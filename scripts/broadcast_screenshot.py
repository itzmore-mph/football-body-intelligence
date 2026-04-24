"""
Broadcast overlay screenshot capture.

Usage:
    python scripts/broadcast_screenshot.py

This script attempts to capture a screenshot of the broadcast overlay
using selenium (preferred) or playwright. If neither is available,
it prints manual capture instructions.

Manual capture instructions:
    1. Run: streamlit run dashboard/broadcast_demo.py
    2. Open http://localhost:8501 in a browser
    3. Set the browser window to 1280x720
    4. Use the browser screenshot tool or OS screenshot utility
    5. Save the result to figures/broadcast_overlay_demo.png
"""


def capture_screenshot(
    url: str = "http://localhost:8501",
    output_path: str = "figures/broadcast_overlay_demo.png",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Capture a screenshot of the broadcast overlay page.

    Attempts selenium first, then playwright. If neither library is
    installed, prints manual capture instructions instead.

    Args:
        url: URL of the running Streamlit app.
        output_path: Destination path for the PNG file.
        width: Viewport width in pixels.
        height: Viewport height in pixels.
    """
    # Try selenium first
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time

        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={width},{height}")

        driver = webdriver.Chrome(options=options)
        try:
            driver.set_window_size(width, height)
            driver.get(url)
            # Allow time for Streamlit to render
            time.sleep(4)
            driver.save_screenshot(output_path)
            print(f"Screenshot saved to {output_path} (via selenium, {width}x{height})")
        finally:
            driver.quit()
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"Selenium capture failed: {exc}")

    # Try playwright as fallback
    try:
        from playwright.sync_api import sync_playwright
        import time

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(url)
            # Allow time for Streamlit to render
            time.sleep(4)
            page.screenshot(path=output_path)
            browser.close()
        print(f"Screenshot saved to {output_path} (via playwright, {width}x{height})")
        return
    except ImportError:
        pass
    except Exception as exc:
        print(f"Playwright capture failed: {exc}")

    # Neither library available: print manual instructions
    print(
        "Neither selenium nor playwright is installed.\n"
        "\n"
        "Manual capture instructions:\n"
        "  1. Run: streamlit run dashboard/broadcast_demo.py\n"
        "  2. Open http://localhost:8501 in a browser\n"
        f"  3. Set the browser window to {width}x{height}\n"
        "  4. Use the browser screenshot tool or OS screenshot utility\n"
        f"  5. Save the result to {output_path}\n"
    )


if __name__ == "__main__":
    capture_screenshot()
