# -*- coding: utf-8 -*-
import sys
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# 표준 출력 인코딩을 UTF-8로 설정
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def crawl_data(url):
    # Chrome 옵션 설정
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920x1080')
    
    # Chrome 드라이버 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        # 페이지 로드
        driver.get(url)
        print("페이지 로드 완료")
        
        # 명시적 대기 추가
        time.sleep(20)
        
        # iframe으로 전환
        try:
            iframe = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "iframe#entryIframe"))
            )
            driver.switch_to.frame(iframe)
            print("iframe 전환 완료")
        except TimeoutException:
            print("iframe을 찾을 수 없습니다.")
            return
        
        # 1. 코스 정보 크롤링
        try:
            course_info = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.J3sor"))
            )
            print("코스 정보 요소 찾음")
            difficulty = course_info.find_element(By.XPATH, ".//div[contains(text(), '난이도')]/following-sibling::div").text
            time_required = course_info.find_element(By.XPATH, ".//div[contains(text(), '소요시간')]/following-sibling::div").text
            course_length = course_info.find_element(By.XPATH, ".//div[contains(text(), '코스길이')]/following-sibling::div").text
            altitude = course_info.find_element(By.XPATH, ".//div[contains(text(), '고도')]/following-sibling::div").text
            
            print("코스 정보:")
            print(f"난이도: {difficulty}")
            print(f"소요시간: {time_required}")
            print(f"코스길이: {course_length}")
            print(f"고도: {altitude}")
        except (TimeoutException, NoSuchElementException) as e:
            print(f"코스 정보를 찾을 수 없습니다: {e}")
        
        # 2. 블로그 리뷰 탭 클릭
        try:
            blog_review_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@href, '/review/ugc')]"))
            )
            driver.execute_script("arguments[0].setAttribute('aria-selected', 'true')", blog_review_tab)
            blog_review_tab.click()
            print("블로그 리뷰 탭 클릭 완료")
            
            # 리뷰 로딩 대기
            time.sleep(15)
        except (TimeoutException, NoSuchElementException) as e:
            print(f"블로그 리뷰 탭을 찾을 수 없습니다: {e}")
        
        # 3. 블로그 리뷰 크롤링
        try:
            review_elements = WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.pcPAT"))
            )
            
            print("\n블로그 리뷰:")
            for element in review_elements:
                try:
                    title = element.find_element(By.CSS_SELECTOR, "div.rGI3P span.place_bluelink").text
                    content = element.find_element(By.CSS_SELECTOR, "div.p7XcC").text
                    print(f"제목: {title}")
                    print(f"내용: {content}")
                    print("---")
                except NoSuchElementException:
                    print("리뷰 요소를 찾을 수 없습니다.")
            
        except Exception as e:
            print(f"블로그 리뷰 크롤링 중 오류 발생: {e}")
        
    except Exception as e:
        print(f"예상치 못한 에러 발생: {e}")
    
    finally:
        driver.quit()

url = "https://map.naver.com/p/search/%EC%84%9C%EC%9A%B8%20%EB%91%98%EB%A0%88%EA%B8%B8%209%EC%BD%94%EC%8A%A4/place/1880081503?c=10.00,0,0,0,dh&placePath=%3Fentry%253Dbmp"

crawl_data(url)