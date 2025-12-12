import requests
import time
import pandas as pd
import logging
from typing import List, Dict, Any


class DistrictCoordinateFetcher:
    """Lớp lấy tọa độ (lat, lon) của phường/xã từ Nominatim API."""

    def __init__(
        self,
        city: str,
        country: str,
        api_url: str = "https://nominatim.openstreetmap.org/search",
        user_agent: str = "Mozilla/5.0 (compatible; MyApp/1.0; +http://example.com/)",
        sleep_time: int = 1,
    ):
        self.city = city
        self.country = country
        self.api_url = api_url
        self.headers = {"User-Agent": user_agent}
        self.sleep_time = sleep_time

    def fetch_coordinate(self, district: str) -> Dict[str, str]:
        """Truy vấn API để lấy tọa độ cho một phường/xã."""
        params = {
            "q": f"{district}, {self.city}, {self.country}",
            "format": "json",
            "addressdetails": 1,
            "limit": 1,
        }

        try:
            response = requests.get(
                self.api_url, params=params, headers=self.headers, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if data:
                lat, lon = data[0]["lat"], data[0]["lon"]
                logging.info(f"[OK] {district}: {lat}, {lon}")
                return {"latitude": lat, "longitude": lon}
            else:
                logging.warning(f"[NOT FOUND] Không tìm thấy tọa độ cho {district}")
                return {}
        except requests.RequestException as e:
            logging.error(f"[ERROR] Truy vấn {district} thất bại: {e}")
            return {}
        finally:
            time.sleep(self.sleep_time)

    def fetch_all(self, districts: List[str]) -> Dict[str, Dict[str, Any]]:
        """Lấy tọa độ cho toàn bộ danh sách phường/xã."""
        coordinates = {}
        for district in districts:
            result = self.fetch_coordinate(district)
            if result:
                coordinates[district] = result
        return coordinates

    def save_to_csv(self, coordinates: Dict[str, Dict[str, Any]], output_file: str):
        """Lưu dữ liệu tọa độ ra file CSV."""
        df = pd.DataFrame(coordinates).T.reset_index()
        df = df.rename(columns={"index": "District"})
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logging.info(f"Đã lưu kết quả vào {output_file}")
        return df


# ===============================
# MAIN
# ===============================
def main():
    # Load từ file CSV crawler trước đó
    input_file = "../raw_data/xa_phuong_TPHCM.csv"
    df_input = pd.read_csv(input_file)

    # Lấy danh sách từ cột "Xã phường mới của TPHCM"
    districts = df_input["Xã phường mới của TPHCM"].dropna().unique().tolist()
    logging.info(f"Đã load {len(districts)} phường/xã từ {input_file}")

    # Fetch tọa độ
    fetcher = DistrictCoordinateFetcher(
        city="Thành phố Hồ Chí Minh",
        country="Việt Nam",
        sleep_time=1
    )

    logging.info("=== Bắt đầu lấy tọa độ ===")
    coordinates = fetcher.fetch_all(districts)

    # Lưu kết quả
    df = fetcher.save_to_csv(coordinates, "../raw_data/toa_do.csv")
    print(df.head())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()