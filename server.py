import web
import os

urls = (
    '/images/(.*)','images', '/', 'index'
)


class MyApplication(web.application):
    def run(self, port=8080, *middleware):
        func = self.wsgifunc(*middleware)
        return web.httpserver.runsimple(func, ('0.0.0.0', port))

class index:
    def GET(self):
        sContent = '<head><meta http-equiv="refresh" content="1"></head>'
        fin = open('result', 'r')
        result=''.join(fin.readlines()).strip()
        sContent += '<img src="./images/{rlt}.png" width="512" height="512">'.format(rlt = result)
        # sContent = '<html><img src="./static/1.png"/></html>'
        return sContent
        
class images:
    def GET(self,name):
        ext = name.split(".")[-1] # Gather extension
        cType = {
            "png":"images/png",
            "jpg":"images/jpeg",
            "gif":"images/gif",
            "ico":"images/x-icon"            }
        if name in os.listdir('C:\\Users\\310149083\\Desktop\\webPyYa\\images'):  # Security
            web.header("Content-Type", cType[ext]) # Set the Header
            return open('images/%s'%name,"rb").read() # Notice 'rb' for reading images
        else:
            raise web.notfound()
if __name__ == "__main__":
    app = MyApplication(urls, globals())
    from web.httpserver import StaticMiddleware
    application = app.wsgifunc(StaticMiddleware)
    app.run(port=3131)