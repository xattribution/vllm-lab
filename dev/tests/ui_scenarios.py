"""Drive the console like a user. Usage: python3 scenario.py NAME  (see SCENARIOS)."""
import asyncio
import os
import sys

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:58120/"
OUT = os.environ.get("SHOTS", "shots") + "/"
os.makedirs(OUT, exist_ok=True)


async def launch_form(pg):
    await pg.evaluate("location.hash='#/launch'")
    await pg.wait_for_timeout(1500)
    await pg.fill("#hubq", "gpt-oss")
    await pg.wait_for_timeout(1200)
    await pg.click("text=openai/gpt-oss-20b")
    await pg.wait_for_timeout(2500)
    await pg.screenshot(path=OUT + "launch-form.png")
    # drag context up and parallel sequences
    await pg.evaluate("""() => { const r = document.querySelector('[data-f=ctx_i]'); r.value = r.max; r.dispatchEvent(new Event('input', {bubbles:true})); }""")
    await pg.evaluate("""() => { const r = document.querySelector('[data-f=seqs]'); r.value = 8; r.dispatchEvent(new Event('input', {bubbles:true})); }""")
    await pg.wait_for_timeout(800)
    await pg.screenshot(path=OUT + "launch-form-big.png")
    await pg.click("summary:has-text('Advanced')")
    await pg.wait_for_timeout(1200)
    await pg.screenshot(path=OUT + "launch-advanced.png", full_page=True)


async def launch_go(pg):
    await launch_form(pg)
    await pg.click("[data-act=form-launch]")
    await pg.wait_for_timeout(1500)
    await pg.screenshot(path=OUT + "launch-after.png")
    await pg.wait_for_timeout(4000)
    await pg.screenshot(path=OUT + "launch-after2.png")


async def conflict(pg):
    await pg.evaluate("location.hash='#/launch?bp=gpt-oss-120b'")
    await pg.wait_for_timeout(3000)
    await pg.screenshot(path=OUT + "conflict-form.png")
    await pg.click("[data-act=form-launch]")
    await pg.wait_for_timeout(1500)
    await pg.screenshot(path=OUT + "conflict-dialog.png")


async def detail(pg):
    name = sys.argv[2] if len(sys.argv) > 2 else "lightning"
    for tab in ("overview", "logs", "connect", "config", "bench", "inspect"):
        await pg.evaluate(f"location.hash='#/engines/{name}/{tab}'")
        await pg.wait_for_timeout(2200)
        await pg.screenshot(path=OUT + f"detail-{tab}.png")


async def play(pg):
    await pg.evaluate("location.hash='#/play'")
    await pg.wait_for_timeout(1500)
    await pg.fill("#pgin", "Explain unified memory in two sentences.")
    await pg.keyboard.press("Enter")
    await pg.wait_for_timeout(1200)
    await pg.screenshot(path=OUT + "play-streaming.png")
    await pg.wait_for_timeout(6000)
    await pg.screenshot(path=OUT + "play-done.png")


async def pages(pg):
    for v in ("library", "hosts", "containers", "webui", "doctor", "settings", "activity", "engines"):
        await pg.evaluate(f"location.hash='#/{v}'")
        await pg.wait_for_timeout(3500 if v in ("doctor", "webui") else 2000)
        await pg.screenshot(path=OUT + f"page-{v}.png")


async def palette(pg):
    await pg.keyboard.press("Control+k")
    await pg.wait_for_timeout(300)
    await pg.keyboard.type("light")
    await pg.wait_for_timeout(300)
    await pg.screenshot(path=OUT + "palette.png")
    await pg.keyboard.press("Escape")
    await pg.click("[data-act=engine-menu] >> nth=0")
    await pg.wait_for_timeout(300)
    await pg.screenshot(path=OUT + "menu.png")


async def ui_flow(pg):
    await pg.evaluate("location.hash='#/deck'")
    await pg.wait_for_timeout(1500)
    await pg.wait_for_timeout(9000)
    btn = pg.locator("[data-key='m-gemma4'] [data-act=start]")
    await btn.click()
    await pg.wait_for_timeout(1200)
    await pg.screenshot(path=OUT + "flow-dialog.png")
    await pg.click("[data-act=conf-go]")
    await pg.wait_for_timeout(2500)
    await pg.screenshot(path=OUT + "flow-booting.png")
    await pg.wait_for_timeout(9000)
    await pg.screenshot(path=OUT + "flow-ready.png")
    await pg.click("[data-key='m-badarg'] [data-act=fix]")
    await pg.wait_for_timeout(1500)
    print("after fix click:", await pg.evaluate("location.hash"))


async def light(pg):
    await pg.evaluate("localStorage.setItem('vllm-lab:theme', JSON.stringify('light'))")
    await pg.reload()
    await pg.wait_for_timeout(2000)
    for v in ("deck", "launch?bp=nemotron-lightning", "engines/lightning"):
        await pg.evaluate(f"location.hash='#/{v}'")
        await pg.wait_for_timeout(2500)
        await pg.screenshot(path=OUT + f"light-{v.split('?')[0].replace('/', '_')}.png")
    await pg.evaluate("localStorage.setItem('vllm-lab:theme', JSON.stringify('dark'))")


async def narrow(pg):
    await pg.set_viewport_size({"width": 820, "height": 1000})
    for v in ("deck", "launch?bp=gpt-oss-20b"):
        await pg.evaluate(f"location.hash='#/{v}'")
        await pg.wait_for_timeout(2500)
        await pg.screenshot(path=OUT + f"narrow-{v.split('?')[0]}.png")


async def more(pg):
    await pg.evaluate("location.hash='#/engines/longctx/config'")
    await pg.wait_for_timeout(2500)
    await pg.evaluate("""() => { const r = document.querySelector('[data-f=util]'); r.value = 0.06; r.dispatchEvent(new Event('input', {bubbles:true})); }""")
    await pg.wait_for_timeout(500)
    await pg.click("[data-act=form-save][data-recreate]")
    await pg.wait_for_timeout(9000)
    await pg.screenshot(path=OUT + "more-config-applied.png")
    await pg.evaluate("location.hash='#/play'")
    await pg.wait_for_timeout(1500)
    await pg.check("[data-act-change=pg-compare]")
    await pg.wait_for_timeout(1500)
    await pg.fill("#pgin", "Hello both")
    await pg.keyboard.press("Enter")
    await pg.wait_for_timeout(5000)
    await pg.screenshot(path=OUT + "more-compare.png")
    await pg.evaluate("location.hash='#/library'")
    await pg.wait_for_timeout(1500)
    await pg.fill("#dlid", "microsoft/phi-4")
    await pg.click("[data-act=lib-dl]")
    await pg.wait_for_timeout(2500)
    await pg.screenshot(path=OUT + "more-download.png")
    await pg.wait_for_timeout(5000)


SCENARIOS = {"more": more, "ui_flow": ui_flow, "light": light, "narrow": narrow, "launch_form": launch_form, "launch_go": launch_go, "conflict": conflict, "detail": detail, "play": play, "pages": pages, "palette": palette}


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch()
        pg = await b.new_page(viewport={"width": 1500, "height": 950})
        errs = []
        pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append(f"PAGEERROR: {e}"))
        await pg.goto(BASE)
        await pg.wait_for_timeout(1500)
        await SCENARIOS[sys.argv[1]](pg)
        print("\n".join(errs) or "no console errors")
        await b.close()

asyncio.run(main())
