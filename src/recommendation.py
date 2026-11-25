"""
recommendation.py - Phân tích đặc điểm thực tế của từng ví để đưa ra khuyến nghị cụ thể,
kèm giải thích rõ cơ sở thống kê từ tập dữ liệu (tại sao chọn từng con số làm mốc).
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.visualization import vi_name, TIME_COLS

TX_ETH_COL = "total transactions (including tnx to create contract"
TX_ERC20_COL = "Total ERC20 tnxs"

# Bảng tổng hợp cơ sở chọn các con số làm mốc (rút từ tập dữ liệu sạch 9.288 ví: 7.632 ví bình thường, 1.656 ví gian lận)
THRESHOLD_BASIS_DATA = [
    {
        "Tiêu chí / Thuộc tính": "Xác suất mô hình (Ngưỡng 30% - 50% - 80%)",
        "Thống kê Ví bình thường (7.632 ví)": "Phần lớn có xác suất dự đoán < 30%",
        "Thống kê Ví gian lận (1.656 ví)": "Phần lớn có xác suất dự đoán >= 50% (nhiều ví > 80%)",
        "Mốc được chọn": "30% (Xem xét) | 50% (Cảnh báo) | 80% (Chặn ngay)",
        "Lý do chọn con số này làm mốc": (
            "Mô hình đã được cân bằng trọng số lớp (class_weight / scale_pos_weight = 4.61) nên 50% là ngưỡng phân lớp chuẩn. "
            "Mốc 30%–50% làm vùng đệm cảnh báo sớm tránh bỏ sót; mốc >= 80% là vùng độ tin cậy rất cao để chặn giao dịch ngay."
        ),
    },
    {
        "Tiêu chí / Thuộc tính": "Tuổi hoạt động của ví (ngày)",
        "Thống kê Ví bình thường (7.632 ví)": "Trung vị: 81.8 ngày (~2.7 tháng)",
        "Thống kê Ví gian lận (1.656 ví)": "Trung vị: 12.3 ngày (54.1% ví gian lận < 15 ngày)",
        "Mốc được chọn": "< 15 ngày (2 tuần)",
        "Lý do chọn con số này làm mốc": (
            "Trung vị tuổi ví gian lận chỉ là 12.3 ngày (thấp hơn 6.6 lần so với 81.8 ngày của ví bình thường). "
            "Chọn mốc < 15 ngày (2 tuần) giúp bao phủ hơn một nửa (54.1%) số ví gian lận dùng ngắn hạn (burner wallet)."
        ),
    },
    {
        "Tiêu chí / Thuộc tính": "ETH nhận trung bình mỗi lần (avg val received)",
        "Thống kê Ví bình thường (7.632 ví)": "Trung vị: 3.76 ETH / lần nhận",
        "Thống kê Ví gian lận (1.656 ví)": "Trung vị: 0.76 ETH / lần nhận (61.4% < 1.0 ETH)",
        "Mốc được chọn": "< 1.0 ETH / lần (khi nhận >= 3 lần)",
        "Lý do chọn con số này làm mốc": (
            "Đây là thuộc tính quan trọng số 1 của mô hình. Trung vị của ví gian lận là 0.76 ETH (< 1 ETH, chiếm 61.4% nhóm gian lận), "
            "phản ánh hành vi thu gom nhiều khoản phí nhỏ lẻ (mint NFT giả, phí nhận airdrop) từ nhiều nạn nhân."
        ),
    },
    {
        "Tiêu chí / Thuộc tính": "Tỷ lệ Số ví gửi đến / Số ví nhận (Gom tiền dạng phễu)",
        "Thống kê Ví bình thường (7.632 ví)": "Trung vị: Nhận từ 2 ví — Gửi đi 2 ví (tỷ lệ 1 : 1)",
        "Thống kê Ví gian lận (1.656 ví)": "Trung vị: Nhận từ 5 ví — Gửi đi 1 ví (tỷ lệ 5 : 1)",
        "Mốc được chọn": "Nhận từ >= 4 ví và Gửi đi <= 2 ví (Nhận >= 2 lần Gửi)",
        "Lý do chọn con số này làm mốc": (
            "Con số >= 4 ví gửi đến cao gấp đôi trung vị ví bình thường (2 ví) và sát trung vị ví gian lận (5 ví); "
            "trong khi chỉ gửi ra <= 2 ví đích phản ánh đúng hành vi gom tiền lừa đảo từ nhiều nguồn rồi rút về 1–2 ví chủ."
        ),
    },
    {
        "Tiêu chí / Thuộc tính": "Tỷ lệ rút cạn số dư ETH",
        "Thống kê Ví bình thường (7.632 ví)": "Trung vị nhận 71.86 ETH, gửi 34.15 ETH (còn giữ tài sản)",
        "Thống kê Ví gian lận (1.656 ví)": "Trung vị nhận 4.22 ETH nhưng số dư còn lại chỉ 0.0012 ETH",
        "Mốc được chọn": "Tổng nhận > 1 ETH, đã rút >= 90%, số dư còn <= 5%",
        "Lý do chọn con số này làm mốc": (
            "Ví gian lận có số dư trung vị gần như bằng 0 (0.0012 ETH, tức rút > 99% tiền nhận được). "
            "Mốc rút >= 90% (giữ lại <= 5%) trên các ví nhận > 1 ETH giúp bắt đúng hành vi ví trung gian tẩu tán tiền ngay sau khi nhận."
        ),
    },
    {
        "Tiêu chí / Thuộc tính": "Hoạt động Token ERC20 (Total ERC20 tnxs)",
        "Thống kê Ví bình thường (7.632 ví)": "Trung vị: 0 giao dịch (chỉ 42.6% ví có ERC20)",
        "Thống kê Ví gian lận (1.656 ví)": "Trung vị: 1 giao dịch (66.8% ví có ERC20, trung vị nhận 1.337 token)",
        "Mốc được chọn": ">= 1 giao dịch ERC20 (kèm xác suất rủi ro >= 30%)",
        "Lý do chọn con số này làm mốc": (
            "Sự khác biệt giữa hai nhóm nằm ngay tại mốc 0 (trung vị ví bình thường) và >= 1 (trung vị ví gian lận, với 66.8% ví gian lận dùng ERC20 "
            "để phát tán token giả hoặc bẫy ký hợp đồng độc hại)."
        ),
    },
    {
        "Tiêu chí / Thuộc tính": "Độ dày lịch sử giao dịch (Tổng giao dịch ETH)",
        "Thống kê Ví bình thường (7.632 ví)": "Trung vị: 11 giao dịch ETH",
        "Thống kê Ví gian lận (1.656 ví)": "Trung vị: 8 giao dịch ETH",
        "Mốc được chọn": "< 5 giao dịch ETH (và < 2 giao dịch ERC20)",
        "Lý do chọn con số này làm mốc": (
            "Mốc < 5 giao dịch thấp hơn một nửa mức trung vị của ví bình thường (11 giao dịch), "
            "cho thấy ví mới chỉ phát sinh vài lệnh đơn lẻ, chưa đủ lịch sử để tin cậy cho giao dịch lớn."
        ),
    },
]

GENERAL_TIPS = [
    (
        "Kiểm tra chống giả mạo địa chỉ (Address Poisoning)",
        "Đối chiếu tối thiểu 6 ký tự đầu và 6 ký tự cuối của địa chỉ ví trước khi ký lệnh chuyển, không sao chép địa chỉ từ lịch sử giao dịch gần nhất.",
    ),
    (
        "Quản lý quyền chi tiêu Token (Token Approvals)",
        "Không cấp quyền chi tiêu không giới hạn (Unlimited Approval) cho hợp đồng lạ; định kỳ kiểm tra và thu hồi quyền tại Etherscan Token Approval Checker hoặc revoke.cash.",
    ),
    (
        "Cảnh giác với Token lạ tự xuất hiện trong ví",
        "Không bán (swap) hoặc truy cập website ghi trong tên các token ERC20 tự nhiên được chuyển vào ví (thủ đoạn Airdrop Phishing).",
    ),
    (
        "Bảo mật khóa cá nhân và chữ ký ngoại tuyến",
        "Tuyệt đối không nhập Seed Phrase / Private Key lên bất kỳ website nào và cẩn trọng với các yêu cầu ký thông điệp (eth_sign / Permit2) không rõ nội dung.",
    ),
]


def show_risk(level, box_type):
    """Hiển thị hộp thông báo mức rủi ro theo màu: success / warning / error."""
    text = f"**Mức rủi ro: {level}**"
    if box_type == "success":
        st.success(text)
    elif box_type == "warning":
        st.warning(text)
    else:
        st.error(text)


def build_case_recommendations(row, level, prob=None):
    """
    Phân tích số liệu thực tế của ví `row` và trả về danh sách các khuyến nghị cụ thể.
    Mỗi phần tử là dict: {"title": ..., "analysis": ..., "basis": ..., "action": ...}.
    """
    age_days = float(row.get("Time Diff between first and last (Mins)", 0)) / 1440.0
    total_tx = float(row.get(TX_ETH_COL, 0))
    sent_tx = float(row.get("Sent tnx", 0))
    rec_tx = float(row.get("Received Tnx", 0))
    uniq_rec = float(row.get("Unique Received From Addresses", 0))
    uniq_sent = float(row.get("Unique Sent To Addresses", 0))
    eth_rec = float(row.get("total ether received", 0))
    eth_sent = float(row.get("total Ether sent", 0))
    eth_bal = float(row.get("total ether balance", 0))
    avg_rec = float(row.get("avg val received", 0))
    erc20_tx = float(row.get(TX_ERC20_COL, 0))
    erc20_rec = float(row.get("ERC20 total Ether received", 0))
    erc20_sent = float(row.get("ERC20 total ether sent", 0))
    contracts = float(row.get("Number of Created Contracts", 0))

    items = []
    prob_pct = f"{prob * 100:.1f}%" if prob is not None else ""

    # 1. Khuyến nghị quyết định giao dịch theo mức độ rủi ro
    if "CAO" in level:
        if prob is not None and prob >= 0.80:
            items.append({
                "title": "Quyết định giao dịch: Chặn / Từ chối giao dịch ngay lập tức",
                "analysis": (
                    f"Mô hình đánh giá xác suất gian lận của ví là {prob_pct} (>= 80%), "
                    "hồ sơ hoạt động trùng khớp rất mạnh với mẫu ví phishing hoặc rửa tiền."
                ),
                "basis": (
                    "Chọn mốc >= 80% làm ngưỡng chặn cứng vì đây là vùng xác suất có độ tin cậy cực cao của mô hình "
                    "(vượt xa ngưỡng phân lớp chuẩn 50%)."
                ),
                "action": (
                    "Dừng toàn bộ lệnh chuyển ETH/token tới địa chỉ này, không ký cấp quyền (Approve/Permit) "
                    "và đưa địa chỉ vào danh sách cảnh báo nội bộ."
                ),
            })
        else:
            items.append({
                "title": "Quyết định giao dịch: Tạm hoãn để xác minh bắt buộc",
                "analysis": (
                    f"Xác suất gian lận của ví là {prob_pct} (nằm trong khoảng cảnh báo 50% – 80%)."
                ),
                "basis": (
                    "Chọn mốc 50% làm ngưỡng kết luận gian lận vì mô hình đã được huấn luyện với trọng số cân bằng lớp "
                    "(scale_pos_weight = 4.61), tại mốc 50% mô hình đạt sự cân bằng tốt nhất giữa Precision và Recall."
                ),
                "action": (
                    "Tạm ngưng chuyển tiền tự động; yêu cầu đối tác xác minh danh tính chủ sở hữu ví qua kênh liên lạc chính thức "
                    "trước khi thực hiện giao dịch."
                ),
            })
    elif level == "CẦN XEM XÉT":
        items.append({
            "title": "Quyết định giao dịch: Giới hạn hạn mức và kiểm tra thủ công",
            "analysis": (
                f"Xác suất gian lận của ví là {prob_pct} (nằm trong vùng nghi vấn 30% – 50%)."
            ),
            "basis": (
                "Chọn khoảng 30% – 50% làm vùng đệm cảnh báo sớm để giảm thiểu nguy cơ bỏ sót các ví gian lận "
                "mấp mé ngưỡng phân lớp 50%."
            ),
            "action": (
                "Nếu vẫn cần giao dịch, chỉ chuyển thử một khoản giá trị nhỏ trước, kiểm tra lịch sử ví trên Etherscan "
                "và chỉ cấp quyền chi tiêu token đúng bằng số lượng giao dịch."
            ),
        })
    else:
        items.append({
            "title": "Quyết định giao dịch: Có thể giao dịch theo quy trình tiêu chuẩn",
            "analysis": (
                f"Xác suất gian lận của ví ở mức thấp ({prob_pct} < 30%), chưa ghi nhận mẫu hành vi lừa đảo điển hình."
            ),
            "basis": (
                "Chọn mốc < 30% là vùng an toàn vì xác suất thấp hơn nhiều so với ngưỡng phân lớp chuẩn 50%."
            ),
            "action": (
                "Có thể tiến hành giao dịch nhưng vẫn cần đối chiếu kỹ địa chỉ ví và lưu ý các đặc điểm hoạt động cụ thể bên dưới."
            ),
        })

    # 2. Phân tích từng trường hợp cụ thể dựa trên số liệu của ví
    # Trường hợp A: Tuổi ví ngắn (< 15 ngày)
    if age_days < 15:
        items.append({
            "title": f"Cảnh báo tuổi hoạt động ngắn ({age_days:,.1f} ngày < mốc 15 ngày)",
            "analysis": (
                f"Thời gian hoạt động từ giao dịch đầu tiên đến cuối cùng của ví chỉ {age_days:,.1f} ngày."
            ),
            "basis": (
                "Trong tập dữ liệu (9.288 ví), ví bình thường có trung vị tuổi hoạt động là 81.8 ngày, "
                "trong khi ví gian lận có trung vị chỉ 12.3 ngày (54.1% ví gian lận có tuổi < 15 ngày). "
                "Do đó mốc < 15 ngày (2 tuần) được chọn để nhận diện ví tạo mới dùng ngắn hạn."
            ),
            "action": (
                "Kiểm tra thời điểm tạo ví trên Etherscan. Nếu đối tác tự giới thiệu là tổ chức/dự án lâu năm nhưng lại dùng ví mới này để nhận tiền thì cần hủy giao dịch."
            ),
        })

    # Trường hợp B: Gom tiền từ nhiều ví nhưng chỉ rút về 1-2 ví đích
    if uniq_rec >= 4 and uniq_sent <= 2 and uniq_rec >= max(uniq_sent * 2, 4):
        items.append({
            "title": f"Dấu hiệu gom tiền dạng phễu (nhận từ {int(uniq_rec)} ví >= 4, chỉ gửi tới {int(uniq_sent)} ví <= 2)",
            "analysis": (
                f"Ví nhận tiền từ {int(uniq_rec)} địa chỉ khác nhau nhưng chỉ chuyển ra cho {int(uniq_sent)} địa chỉ đích."
            ),
            "basis": (
                "Trong tập dữ liệu, ví bình thường có trung vị Nhận từ 2 ví và Gửi đi 2 ví (tỷ lệ 1:1). "
                "Ngược lại, ví gian lận có trung vị Nhận từ 5 ví nhưng chỉ Gửi đi 1 ví (tỷ lệ 5:1). "
                "Vì vậy chọn mốc Nhận >= 4 ví (gấp đôi trung vị ví sạch) và Gửi <= 2 ví để bắt mẫu gom tiền lừa đảo."
            ),
            "action": (
                "Mở tab giao dịch gửi đi (Outbound) của ví trên Etherscan để kiểm tra địa chỉ nhận cuối cùng xem có bị gắn nhãn Phish/Hack hoặc chuyển vào máy trộn tiền (Mixer) không."
            ),
        })

    # Trường hợp C: Ví trung gian rút cạn số dư
    if eth_rec > 1.0 and eth_sent >= 0.9 * eth_rec and abs(eth_bal) <= 0.05 * eth_rec:
        items.append({
            "title": f"Hành vi rút cạn số dư (nhận {eth_rec:,.2f} ETH, đã rút {eth_sent:,.2f} ETH >= 90%)",
            "analysis": (
                f"Ví hầu như không lưu giữ tài sản (số dư còn lại {eth_bal:,.4f} ETH <= 5% tổng nhận)."
            ),
            "basis": (
                "Trong tập dữ liệu, ví gian lận có trung vị tổng nhận 4.22 ETH nhưng số dư trung vị còn lại chỉ 0.0012 ETH (rút > 99%). "
                "Chọn mốc tổng nhận > 1 ETH nhưng đã rút >= 90% (còn lại <= 5%) để nhận diện ví trung gian tẩu tán tiền nhanh."
            ),
            "action": (
                "Tránh chuyển khoản giá trị lớn hoặc đặt cọc vào địa chỉ này vì dòng tiền bị rút đi ngay lập tức, rủi ro không thể thu hồi rất cao."
            ),
        })

    # Trường hợp D: Nhận nhiều khoản ETH nhỏ lẻ (< 1.0 ETH/lần)
    if rec_tx >= 3 and 0 < avg_rec < 1.0 and ("CAO" in level or level == "CẦN XEM XÉT"):
        items.append({
            "title": f"Mô hình thu nhiều khoản ETH nhỏ lẻ (trung bình {avg_rec:,.3f} ETH/lần < mốc 1.0 ETH)",
            "analysis": (
                f"Qua {int(rec_tx)} lần nhận (>= 3 lần), mỗi giao dịch chỉ trung bình {avg_rec:,.3f} ETH."
            ),
            "basis": (
                "Thuộc tính 'avg val received' có độ quan trọng số 1 trong mô hình. Ví bình thường có trung vị 3.76 ETH/lần, "
                "còn ví gian lận có trung vị chỉ 0.76 ETH/lần (61.4% ví gian lận < 1.0 ETH/lần). "
                "Do đó chọn mốc < 1.0 ETH/lần (lặp lại >= 3 lần) làm ngưỡng phát hiện thu gom tiền nhỏ lẻ."
            ),
            "action": (
                "Cảnh giác với các hình thức thu phí nhỏ lẻ từ nhiều nạn nhân như mint NFT giả, phí mở khóa thưởng airdrop hoặc phí xác minh ví."
            ),
        })

    # Trường hợp E: Có giao dịch token ERC20 đáng ngờ
    if erc20_tx > 0 or erc20_rec > 0 or erc20_sent > 0:
        if "CAO" in level or level == "CẦN XEM XÉT":
            items.append({
                "title": f"Rủi ro từ hoạt động Token ERC20 ({int(erc20_tx)} giao dịch >= mốc 1 giao dịch)",
                "analysis": (
                    f"Ví có {int(erc20_tx)} giao dịch ERC20 (tổng nhận {erc20_rec:,.2f}, tổng gửi {erc20_sent:,.2f} token)."
                ),
                "basis": (
                    "Trong tập dữ liệu, trung vị số giao dịch ERC20 của ví bình thường là 0 (chỉ 42.6% ví có ERC20), "
                    "trong khi trung vị của ví gian lận là 1 giao dịch (66.8% ví gian lận có hoạt động ERC20). "
                    "Vì vậy mốc >= 1 giao dịch ERC20 đi kèm xác suất rủi ro >= 30% được dùng để cảnh báo lừa đảo qua token."
                ),
                "action": (
                    "Kiểm tra mục ERC20 Token Txns trên Etherscan; không tương tác với các token lạ gắn liền với ví này và thu hồi ngay quyền Approve nếu đã từng kết nối ví."
                ),
            })

    # Trường hợp F: Ví có triển khai Smart Contract
    if contracts > 0:
        items.append({
            "title": f"Ví có triển khai hợp đồng thông minh ({int(contracts)} Smart Contract >= 1)",
            "analysis": (
                f"Địa chỉ này đã trực tiếp khởi tạo {int(contracts)} hợp đồng trên mạng Ethereum."
            ),
            "basis": (
                "Hơn 90% ví cá nhân thông thường trong dữ liệu có số hợp đồng tạo mới bằng 0. "
                "Khi số hợp đồng >= 1, ví đóng vai trò người triển khai mã lệnh (Deployer) nên cần kiểm tra độ an toàn của mã nguồn."
            ),
            "action": (
                "Trước khi tương tác với hợp đồng do ví này tạo, cần kiểm tra tab Contract trên Etherscan xem mã nguồn đã được xác thực (Verified) hay chưa."
            ),
        })

    # Trường hợp G: Lịch sử giao dịch quá mỏng
    if total_tx < 5 and erc20_tx < 2:
        items.append({
            "title": f"Lịch sử giao dịch rất mỏng ({int(total_tx)} giao dịch ETH < mốc 5 giao dịch)",
            "analysis": (
                f"Ví mới chỉ thực hiện {int(total_tx)} giao dịch ETH (gửi {int(sent_tx)}, nhận {int(rec_tx)})."
            ),
            "basis": (
                "Trung vị tổng giao dịch ETH của ví bình thường trong dữ liệu là 11 giao dịch. "
                "Chọn mốc < 5 giao dịch (thấp hơn một nửa mức trung vị) để đánh dấu các ví chưa đủ lịch sử kiểm chứng uy tín."
            ),
            "action": (
                "Nên yêu cầu giao dịch thử với số tiền nhỏ hoặc đề nghị đối tác sử dụng địa chỉ ví chính có lịch sử hoạt động rõ ràng hơn."
            ),
        })

    # Trường hợp H: Nếu ví rủi ro thấp và hoạt động ổn định lâu dài
    if level == "THẤP" and len(items) == 1:
        items.append({
            "title": f"Hồ sơ hoạt động ổn định ({age_days:,.1f} ngày >= 15 ngày, {int(total_tx):,} giao dịch ETH >= 5)",
            "analysis": (
                f"Ví có thời gian hoạt động dài ({age_days:,.1f} ngày), giao dịch hai chiều cân đối "
                f"(gửi {int(sent_tx):,} lần, nhận {int(rec_tx):,} lần từ {int(uniq_rec):,} địa chỉ)."
            ),
            "basis": (
                "Các chỉ số của ví đều vượt ngưỡng an toàn và nằm gần trung vị của nhóm ví bình thường trong tập huấn luyện "
                "(tuổi ví trung vị 81.8 ngày, tổng giao dịch trung vị 11 lần)."
            ),
            "action": (
                "Duy trì quy tắc an toàn cơ bản: kiểm tra kỹ 6 ký tự đầu và cuối của địa chỉ nhận trước khi xác nhận giao dịch trên ví."
            ),
        })

    return items


def summarize_case_for_csv(row, level, prob=None):
    """Tạo chuỗi khuyến nghị ngắn gọn, sát thực tế từng ví để xuất ra bảng/file CSV."""
    items = build_case_recommendations(row, level, prob)
    if len(items) >= 2:
        return f"{items[0]['title']}. {items[1]['title']} — {items[1]['action']}"
    return f"{items[0]['title']} — {items[0]['action']}"


def render_recommendations_box(row, level, prob=None):
    """Hiển thị các khuyến nghị trong khung đóng hộp (border=True), không dùng dấu chấm đầu dòng."""
    items = build_case_recommendations(row, level, prob)
    with st.container(border=True):
        for idx, item in enumerate(items, start=1):
            if idx > 1:
                st.divider()
            st.markdown(f"**{idx}. {item['title']}**")
            st.caption(f"Phân tích hồ sơ ví: {item['analysis']}")
            st.write(f"Hướng xử lý: {item['action']}")


def compare_with_groups(row, df_train, features, target_col="FLAG"):
    """
    So sánh giá trị của 1 ví với trung vị của ví bình thường và ví gian lận (tập Train).
    Với mỗi thuộc tính, cho biết ví đang GẦN nhóm nào hơn (so trên thang log).
    """
    med_normal = df_train[df_train[target_col] == 0][features].median()
    med_fraud = df_train[df_train[target_col] == 1][features].median()

    rows = []
    for f in features:
        v, n, fr = float(row[f]), float(med_normal[f]), float(med_fraud[f])
        unit = 1440.0 if f in TIME_COLS else 1.0  # phút -> ngày
        d_normal = abs(np.log1p(abs(v)) - np.log1p(abs(n)))
        d_fraud = abs(np.log1p(abs(v)) - np.log1p(abs(fr)))
        rows.append({
            "Thuộc tính": vi_name(f),
            "Giá trị của ví": round(v / unit, 3),
            "Trung vị ví bình thường": round(n / unit, 3),
            "Trung vị ví gian lận": round(fr / unit, 3),
            "Gần nhóm": "Gian lận" if d_fraud < d_normal else "Bình thường",
        })
    return pd.DataFrame(rows)
