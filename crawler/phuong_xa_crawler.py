from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import pandas as pd

def crawl_table(url, output_csv="../raw_data/xa_phuong_TPHCM.csv"):
    print(f"🔗 Đang gửi request tới: {url}")
    req = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            )
        },
    )
    html = urlopen(req)
    print("✅ Đã nhận phản hồi từ server")

    bs = BeautifulSoup(html.read(), "html.parser")
    print("✅ Đã parse HTML bằng BeautifulSoup")

    # Tìm bảng
    table = bs.find("article").find("table")
    if not table:
        raise ValueError("❌ Không tìm thấy bảng trong trang web!")

    print("✅ Đã tìm thấy bảng dữ liệu")

    # Header
    headers = [td.get_text(strip=True) for td in table.find("tr").find_all("td")]
    print(f"📝 Header: {headers}")

    # Rows
    rows = []
    for tr in table.find_all("tr")[1:]:
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:
            rows.append(cells)

    print(f"📊 Đã thu thập {len(rows)} dòng dữ liệu")

    # DataFrame
    df = pd.DataFrame(rows, columns=headers)
    print("✅ Đã tạo DataFrame")

    # Lưu CSV
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"💾 Đã lưu {len(df)} dòng vào file: {output_csv}")

    return df


if __name__ == "__main__":
    url = "https://thuvienphapluat.vn/phap-luat/ho-tro-phap-luat/tra-cuu-168-phuong-xa-tphcm-chinh-thuc-sau-sap-nhap-nam-2025-day-du-chi-tiet-danh-sach-toan-bo-phuo-570655-223893.html?fbclid=IwY2xjawNHPDpleHRuA2FlbQIxMABicmlkETExSndURmp1b3RSeXk4TExXAR4qggofLBHN1wySsLWjhI9D2sHoqZRNz4rQOAWmMzS7vndBE51LdU6C_knpHQ_aem_4-n2wRd0fsqe4rCzOL9zSw"
    df = crawl_table(url)
