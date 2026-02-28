"""
Run this ONCE to permanently fix the DNS error on Windows.
It uninstalls aiodns, which causes aiohttp to always use
the Windows system resolver (ThreadedResolver) instead.
"""
import subprocess
import sys

print("Fixing Windows DNS issue...")
print("Uninstalling aiodns (causes DNS failures on Windows)...")

result = subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "aiodns", "-y"],
    capture_output=True, text=True
)

if "Successfully uninstalled" in result.stdout:
    print("✅ aiodns uninstalled successfully!")
    print("✅ Now run: python bot.py")
elif "not installed" in result.stderr or "WARNING" in result.stderr:
    print("ℹ️  aiodns was not installed (or already removed).")
    print("   The DNS error might be caused by something else.")
    print("   Try running bot.py anyway — it may work now.")
else:
    print("Output:", result.stdout)
    print("Stderr:", result.stderr)

input("\nPress Enter to exit...")
