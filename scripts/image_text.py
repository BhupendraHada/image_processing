from PIL import Image, ImageDraw

from pytesseract import image_to_string, image_to_pdf_or_hocr


img = Image.open('/home/user/Downloads/5bbec42e4b226-20181011090158.jpg')

text = image_to_string(img)
box = image_to_pdf_or_hocr(img)
print box

f = open('/home/user/Downloads/text_imgage.png', "wb")

img = Image.new('RGB', (500, 500), color=(73, 109, 137))

d = ImageDraw.Draw(img)
d.text((50, 50), text.encode("utf-8"), fill=(255, 255, 0))

img.save('/home/user/Downloads/text_imgage.png')
