import asyncio
from playwright.async_api import async_playwright

async def verify():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Log console browser ke terminal kita
        page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
        
        print("Navigasi ke http://127.0.0.1:45678 ...")
        await page.goto("http://127.0.0.1:45678")
        
        print("Menunggu 5 detik untuk rendering...")
        await asyncio.sleep(5)
        
        # Ambil screenshot
        await page.screenshot(path="_LOG/verify_45678.png")
        print("Screenshot disimpan di _LOG/verify_45678.png")
        
        # Cek apakah elemen chart ada
        chart_exists = await page.query_selector("#chart")
        print(f"Elemen #chart ditemukan: {chart_exists is not None}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify())
