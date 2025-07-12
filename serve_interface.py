from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import webbrowser
import threading
import time

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def open_browser():
    """فتح المتصفح بعد ثانيتين"""
    time.sleep(2)
    webbrowser.open('http://localhost:8080/search_interface.html')

def main():
    # التأكد من وجود ملف HTML
    if not os.path.exists('search_interface.html'):
        print("❌ ملف search_interface.html غير موجود!")
        return
    
    # تشغيل الخادم
    port = 8080
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    
    print(f"🚀 تم تشغيل الخادم على المنفذ {port}")
    print(f"🌐 افتح المتصفح على: http://localhost:{port}/search_interface.html")
    print("⏹️  اضغط Ctrl+C لإيقاف الخادم")
    
    # فتح المتصفح تلقائياً
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n⏹️  تم إيقاف الخادم")
        httpd.server_close()

if __name__ == '__main__':
    main() 