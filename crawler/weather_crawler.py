import requests
import pandas as pd
import time
import logging
import os
import calendar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

WEATHER_BIT_API_KEY = ""
INPUT_FILE = "xa_phuong_city_name_TPHCM.csv"
OUTPUT_FILE_TEMPLATE = "weather_city_{year}.csv"
SLEEP_TIME = 1

BASE_URL = "https://api.weatherbit.io/v2.0/history/hourly"


def get_date_ranges(year: int):
    """Sinh ra list (start_date, end_date) theo từng tháng của năm"""
    ranges = []
    for month in range(1, 9 + 1):
        start_date = f"{year}-{month:02d}-01"
        last_day = calendar.monthrange(year, month)[1]  # ngày cuối tháng
        end_date = f"{year}-{month:02d}-{last_day:02d}"
        ranges.append((start_date, end_date))
    return ranges


def fetch_air_quality(city_name, start_date, end_date):
    url = f"{BASE_URL}?city={city_name}&start_date={start_date}&end_date={end_date}&key={WEATHER_BIT_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            logging.error(f"HTTP 429 cho {city_name} ({start_date} → {end_date}) - bị giới hạn rate limit")
            time.sleep(60)  # chờ rồi thử tiếp
            return None
        else:
            logging.error(f"HTTP {response.status_code} cho {city_name} ({start_date} → {end_date})")
            return None
    except Exception as e:
        logging.error(f"Lỗi khi gọi API cho {city_name}: {e}")
        return None


def crawl_air_quality(year):
    districts = pd.read_csv(INPUT_FILE)

    # Tạo file output cho năm đó
    OUTPUT_FILE = OUTPUT_FILE_TEMPLATE.format(year=year)

    # Nếu có dữ liệu cũ thì lấy danh sách (city_name, start_date, end_date)
    if os.path.exists(OUTPUT_FILE):
        done_data = pd.read_csv(OUTPUT_FILE)
        if {"city_name", "start_date", "end_date"}.issubset(done_data.columns):
            done_pairs = set(zip(done_data["city_name"], done_data["start_date"], done_data["end_date"]))
        else:
            done_pairs = set()
    else:
        done_pairs = set()

    logging.info(f"Đọc {len(districts)} dòng từ {INPUT_FILE}")

    for city_name in districts["city_name"].unique():
        logging.info(f"▶️ Bắt đầu crawl {city_name} ({year})")

        for start_date, end_date in get_date_ranges(year):
            key = (city_name, start_date, end_date)

            # Bỏ qua nếu đã crawl tháng này
            if key in done_pairs:
                logging.info(f"⏩ Bỏ qua {city_name} [{start_date} → {end_date}] (đã crawl trước đó)")
                continue

            try:
                logging.info(f"▶️ Crawl {city_name} ({start_date} → {end_date})")
                data = fetch_air_quality(city_name, start_date, end_date)

                if data and "data" in data:
                    records = data["data"]
                    for r in records:
                        r["city_name"] = city_name
                        r["start_date"] = start_date
                        r["end_date"] = end_date

                    # ⬇️ Lưu ngay tháng này vào CSV
                    pd.DataFrame(records).to_csv(
                        OUTPUT_FILE, mode="a", header=not os.path.exists(OUTPUT_FILE), index=False
                    )

                    logging.info(f"✅ Lưu {city_name} ({len(records)} bản ghi) [{start_date} → {end_date}]")
                else:
                    logging.warning(f"⚠️ Không có dữ liệu cho {city_name} [{start_date} → {end_date}]")

            except Exception as e:
                logging.error(f"Lỗi khi crawl {city_name} ({start_date} → {end_date}): {e}")

            time.sleep(SLEEP_TIME)


if __name__ == "__main__":
    year = 2025
    crawl_air_quality(year)
