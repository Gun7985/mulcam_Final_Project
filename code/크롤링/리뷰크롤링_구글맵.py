# -*- coding: utf-8 -*-
import sys
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# 표준 출력 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def crawl_review(url):
    # Chrome 옵션 설정
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')  # GUI 없이 실행
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    # Chrome 드라이버 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # 페이지 로드
        driver.get(url)
        
        # 리뷰 요소가 로드될 때까지 대기 (최대 10초)
        review_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "wiI7pd"))
        )
        
        # 리뷰 텍스트 추출
        review_text = review_element.text
        return review_text
    
    except Exception as e:
        print(f"에러 발생: {e}")
        return "리뷰를 찾을 수 없습니다."
    
    finally:
        driver.quit()

url = "https://www.google.com/maps/place/%EC%84%9C%EC%9A%B8%EB%91%98%EB%A0%88%EA%B8%B84-2%EC%BD%94%EC%8A%A4(%EB%A7%A4%ED%97%8C%EC%8B%9C%EB%AF%BC%EC%9D%98%EC%88%B2~%EC%82%AC%EB%8B%B9%EC%97%AD)/data=!3m1!1e3!4m12!1m2!2m1!1z7ISc7Jq4IOuRmOugiOq4uCA07L2U7Iqk!3m8!1s0x357ca1589596e84b:0xbea6875e2dbaaf44!8m2!3d37.4715027!4d127.0329115!9m1!1b1!15sChjshJzsmrgg65GY66CI6ri4IDTsvZTsiqSSAQtoaWtpbmdfYXJlYeABAA!16s%2Fg%2F11nx2w2zv_?entry=ttu"

review = crawl_review(url)
print("\nCrawled Review:")
print(review)