# Author: S. Kato (Graduate School of Information Technology and Science, Osaka University)
# Version: 3.0
# License: MIT License

# cython: linetrace=True
import cython
from datetime import datetime
from loguru import logger
import re

import numpy as np
import pyshark
from tqdm import tqdm

cimport numpy as cnp
from libc.stdint cimport uint8_t

ctypedef cnp.float64_t DOUBLE
ctypedef cnp.int64_t INT64

cdef double PI = np.pi


hex_to_bin = str.maketrans(
    {
        "0": "0000",
        "1": "0001",
        "2": "0010",
        "3": "0011",
        "4": "0100",
        "5": "0101",
        "6": "0110",
        "7": "0111",
        "8": "1000",
        "9": "1001",
        "a": "1010",
        "b": "1011",
        "c": "1100",
        "d": "1101",
        "e": "1110",
        "f": "1111",
    }
)


def get_v_matrix(pcap_file, address, num_to_process=None, verbose=False, validate_unitary=False):
    """
    Extract V-matrices from a PCAP/PCAPNG containing VHT/HE compressed beamforming reports.

    Notes
    - Uses tshark/pyshark for parsing, which can be slow on large captures.
    - `num_to_process` limits how many matching packets are decoded (useful for tests/smoke runs).
    - Always closes the underlying capture to avoid leaking `tshark` processes.
    """

    # IMPORTANT:
    # Don't slice fixed offsets out of `frame_raw` (radiotap header length varies).
    # Instead, rely on Wireshark's decoded management fields (wlan.mgt.*),
    # which are stable across link-layer encapsulations.
    cap = pyshark.FileCapture(
        pcap_file,
        # "Action No Ack" management frames are subtype 0x000e in Wireshark.
        # These frames commonly carry VHT/HE compressed beamforming feedback.
        display_filter=f"wlan.fc.type_subtype == 0x000e and wlan.ta == {address}",
        use_json=False,
        include_raw=False,
        keep_packets=False,
    )
    p = cap._packets_from_tshark_sync()

    # parameter setting
    phi_psi_matching = [(4.0, 2.0), (6.0, 4.0)]

    # sequentially process packets
    ts = []
    vs = []
    p_cnt = 0

    try:
        while True:
            try:
                packet = p.__next__()
            except StopIteration:
                break

            p_cnt += 1
            if num_to_process is not None and p_cnt > num_to_process:
                break
            if verbose:
                logger.info(f"parsing {p_cnt} packets...")

            # timestamp
            try:
                timestamp = float(packet.frame_info.time_epoch)
            except Exception:
                # Handle ISO format like 2022-07-06T09:43:17.826467072Z
                ts_str = str(packet.frame_info.time_epoch)
                if ts_str.endswith("Z"):
                    ts_str = ts_str[:-1]
                if "." in ts_str:
                    left, frac = ts_str.split(".", 1)
                    # Keep at most microseconds for fromisoformat compatibility
                    frac = re.sub(r"[^0-9].*$", "", frac)  # strip any timezone suffixes
                    frac = (frac + "000000")[:6]
                    ts_str = f"{left}.{frac}"
                dt = datetime.fromisoformat(ts_str)
                timestamp = dt.timestamp()

            # Pull decoded management fields (preferred).
            # Some packets may not decode as wlan.mgt; skip those.
            try:
                mgt = packet["wlan.mgt"]
            except Exception:
                if verbose:
                    logger.warning("Packet missing wlan.mgt layer; skipping.")
                continue

            try:
                category_code = int(mgt.wlan_fixed_category_code)
            except Exception:
                if verbose:
                    logger.warning("Missing/invalid wlan_fixed_category_code; skipping.")
                continue

            # check VHT or HE
            if category_code == 21:
                # VHT compressed beamforming report
                try:
                    codebook_info = int(str(mgt.wlan_vht_mimo_control_codebookinfo), 16)
                    bw = int(str(mgt.wlan_vht_mimo_control_chanwidth), 16)
                    nr = int(str(mgt.wlan_vht_mimo_control_nrindex), 16) + 1
                    nc = int(str(mgt.wlan_vht_mimo_control_ncindex), 16) + 1
                    cbr_hex = str(mgt.wlan_vht_compressed_beamforming_report)
                except Exception as e:
                    if verbose:
                        logger.warning(f"VHT fields missing/invalid; skipping. err={e}")
                    continue
            elif category_code == 30:
                # HE compressed beamforming report
                # Not all tshark builds expose the same HE field names; for now we require
                # the HE MIMO control and compressed matrices field to be present.
                try:
                    he_mimo_control = str(mgt.wlan_he_action_he_mimo_control)
                    he_mimo_control_bin = bin(int(he_mimo_control, 16))[2:].zfill(40)
                    ru_end_index = int(he_mimo_control_bin[11:17], 2)
                    ru_start_index = int(he_mimo_control_bin[17:23], 2)
                    codebook_info = int(he_mimo_control_bin[30], 2)
                    bw = int(he_mimo_control_bin[32:34], 2)
                    nr = int(he_mimo_control_bin[34:37], 2) + 1
                    nc = int(he_mimo_control_bin[37:], 2) + 1
                    # Best-effort: use the generic compressed BF matrices bytes if available.
                    # This may differ from the legacy raw slicing format.
                    cbr_hex = str(mgt.wlan_mimo_csimatrices_cbf)
                except Exception as e:
                    if verbose:
                        logger.warning(f"HE fields missing/invalid; skipping. err={e}")
                    continue
            else:
                if verbose:
                    logger.warning(f"Unknown category code: {category_code}. Skipping packet.")
                continue

            # Normalize hex payload (pyshark sometimes inserts ':' separators).
            cbr_hex = cbr_hex.replace(":", "").replace(" ", "").lower()

            num_snr = nc
            (phi_size, psi_size) = phi_psi_matching[codebook_info]

            # calc binary splitting rule
            angle_bits_order = []
            angle_type = []
            angle_index = []
            phi_indices = [0, 0]
            psi_indices = [1, 0]

            angle_bits_order_len = min([nc, nr - 1]) * (2 * (nr - 1) - min(nc, nr - 1) + 1)
            if angle_bits_order_len == 0:
                if verbose:
                    logger.warning(f"angle_bits_order_len is 0 (nr={nr}, nc={nc}). Skipping packet.")
                continue
            cnt = nr - 1
            while len(angle_bits_order) < angle_bits_order_len:
                for i in range(cnt):
                    angle_bits_order.append(phi_size)
                    angle_type.append("phi")
                    angle_index.append([phi_indices[0] + i, phi_indices[1]])
                phi_indices[0] += 1
                phi_indices[1] += 1
                for i in range(cnt):
                    angle_bits_order.append(psi_size)
                    angle_type.append("psi")
                    angle_index.append([psi_indices[0] + i, psi_indices[1]])
                psi_indices[0] += 1
                psi_indices[1] += 1
                cnt -= 1

            num_subc = int(
                (len(cbr_hex) - num_snr * 2)
                * 4
                // (phi_size * angle_type.count("phi") + psi_size * angle_type.count("psi"))
            )

            split_rule = np.zeros(angle_bits_order_len + 1)
            split_rule[1:] = np.cumsum(angle_bits_order)
            split_rule = split_rule.astype(np.int32)
            angle_seq_len = split_rule[-1]
            cbr, snr = hex_to_quantized_angle(
                cbr_hex, num_snr, num_subc, angle_seq_len, split_rule
            )

            # V matrix recovery
            v = np.zeros((num_subc, nr, nc), dtype=complex)
            subc_len = len(angle_type)
            for subc in range(num_subc):
                angle_slice = cbr[subc * subc_len : (subc + 1) * subc_len]
                angle_slice = [
                    quantized_angle_formulas(t, a, phi_size, psi_size)
                    for t, a in zip(angle_type, angle_slice)
                ]
                mat_e = inverse_givens_rotation(
                    nr, nc, angle_slice, angle_type, angle_index
                )
                v[subc] = mat_e

                if validate_unitary:
                    # expensive; keep off by default
                    if not np.all((np.sum(np.abs(mat_e) ** 2, axis=0) - 1) < 1e-5):
                        raise ValueError(
                            f"V is not unitary: {np.sum(np.abs(mat_e) ** 2, axis=0)}"
                        )

            # Check for consistency with previous packets
            if len(vs) > 0 and v.shape != vs[0].shape[1:]:
                if verbose:
                    logger.warning(
                        f"Packet shape mismatch: {v.shape} vs {vs[0].shape[1:]}. Skipping."
                    )
                continue

            vs.append(v[np.newaxis])
            ts.append(timestamp)
    finally:
        try:
            cap.close()
        except Exception:
            pass

    if len(vs) == 0:
        # Keep return types stable even if no packets matched the filter.
        return np.array([], dtype=np.float64), np.empty((0, 0, 0, 0), dtype=complex)

    vs = np.concatenate(vs)
    ts = np.array(ts)

    if verbose:
        logger.info(f"{ts.shape[0]} packets are parsed.")

    return ts, vs

cdef hex_to_quantized_angle(
    str cbr_hex,
    int num_snr,
    int num_subc,
    int angle_seq_len,
    cnp.ndarray[int] split_rule,
):
    cdef:
        str cbr_bin, snr_bin
        list cbr_subc_split, angle_dec, hex_split
        list cbr = []
        list snr = []
        int snr_idx, i, start, max_length
        cnp.ndarray[int] angle_bits_order

    cbr_bin = cbr_hex.translate(hex_to_bin)[::-1]

    for i in range(num_snr):
        snr_bin = cbr_bin[i * 8 : (i + 1) * 8][::-1]
        if snr_bin[0] == "0":
            snr_idx = <int>int(snr_bin, 2)
        else:
            snr_idx = -(<int>int(snr_bin, 2) ^ 0b11111111)
        snr.append(-(-128 - snr_idx) * 0.25 - 10)

    cbr_bin = cbr_bin[num_snr * 8 :]
    max_length = num_subc * angle_seq_len
    angle_bits_order = split_rule[1:] - split_rule[:-1]

    for s in [cbr_bin[i : i + angle_seq_len] for i in range(0, max_length, angle_seq_len)]:
        if len(s) != split_rule[-1]:
            continue
        angle_dec = [None] * (len(angle_bits_order) - 1)
        start = 0
        for i in range(1, len(angle_bits_order)):
            angle_dec[i - 1] = <int>int(s[start : start + angle_bits_order[i]], 2)
            start += angle_bits_order[i]

        cbr.extend(angle_dec)

    return cbr, snr


cdef inverse_givens_rotation(int nrx, int ntx, list angles, list angle_types, list angle_indices):
    cdef:
        cnp.ndarray[complex, ndim=2] mat_e = np.eye(N=nrx, M=ntx, dtype=complex)
        cnp.ndarray[complex, ndim=2] d_li = np.eye(N=nrx, M=nrx, dtype=complex)
        cnp.ndarray[complex, ndim=2] g_li = np.eye(nrx, nrx, dtype=complex)
        INT64 d_count = 0
        INT64 d_patience = 1
        INT64 idx
        str a_t
        list a_i
        DOUBLE cos_val, sin_val

    reverse_mat = []
    for idx in reversed(range(len(angles))):
        a_t = angle_types[idx]
        a_i = angle_indices[idx]

        if a_t == "phi":
            d_li[a_i[0], a_i[0]] = np.exp(1j * angles[idx])
            d_count += 1
        elif a_t == "psi":
            cos_val = np.cos(angles[idx])
            sin_val = np.sin(angles[idx])
            g_li[a_i[1], a_i[1]] = cos_val
            g_li[a_i[1], a_i[0]] = sin_val
            g_li[a_i[0], a_i[1]] = -sin_val
            g_li[a_i[0], a_i[0]] = cos_val
            mat_e = g_li.T @ mat_e
            g_li = np.eye(nrx, nrx, dtype=complex)
        else:
            raise ValueError("inverse_givens_rotation(): invalid angle type")
        
        if d_count == d_patience:
            mat_e = d_li.T @ mat_e
            d_patience += 1
            d_count = 0
            d_li = np.eye(nrx, nrx, dtype=complex)

    return mat_e


cdef quantized_angle_formulas(str angle_type, int angle, int phi_size, int psi_size):
    angle_funcs = {
        "phi": lambda x: PI * x / (2.0 ** (phi_size - 1.0)) + PI / (2.0 ** (phi_size)),
        "psi": lambda x: PI * x / (2.0 ** (psi_size + 1.0))
        + PI / (2.0 ** (psi_size + 2.0)),
    }
    return angle_funcs[angle_type](angle)


cdef hex_flip(str hex_str):
    return "".join(reversed([hex_str[i : i + 2] for i in range(0, len(hex_str), 2)]))
