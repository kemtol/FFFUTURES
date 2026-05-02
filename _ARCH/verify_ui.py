import asyncio
from playwright.async_api import async_playwright
import os

async def run():
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch()
        page = await browser.new_page()
        print("Navigating to http://127.0.0.1:8080 ...")
        await page.goto('http://127.0.0.1:8080')
        
        print("Waiting for chart to load (5s)...")
        await asyncio.sleep(5)
        
        screenshot_path = '_LOG/backtest_preview.png'
        await page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        
        # Cek apakah ada error di console browser
        # (Bisa dikembangkan nanti)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())
