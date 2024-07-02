import fitz  # PyMuPDF

"""
 pdf에서 toc를 추출, toc를 인식하여 chapter별로 텍스트를 추출
"""

def extract_toc(pdf_path):
    pdf_document = fitz.open(pdf_path)
    toc = pdf_document.get_toc()
    for i, entry in enumerate(toc):
        toc[i] = (entry[0], entry[1].replace(":", "").replace("  ", " "), entry[2])
    return toc


def extract_text_by_toc(pdf_path, toc, header_px, footer_px):
    pdf_document = fitz.open(pdf_path)
    toc_texts = []
    for i, entry in enumerate(toc):
        level, start_title, start_page = entry
        print(i, 'i\n')
        print(len(toc), 'len(toc)\n')
        if i + 1 < len(toc):
            end_page = toc[i + 1][2]
            end_title = toc[i + 1][1]
        else:
            end_page = pdf_document.page_count
            end_title = 1
        page_text = extract_text_excluding_header_footer(
            pdf_document,
            start_page,
            end_page,
            header_px,
            footer_px,
            start_title,
            end_title,
        )
        toc_texts.append((start_title, page_text))
        # print("-------------------")
        # print(title, level, start_page, end_page, "\n", page_text)
        save_text_to_file(start_title, page_text, level, start_page, end_page)
    return toc_texts


def extract_text_excluding_header_footer(
    pdf_document, start_page, end_page, header_px, footer_px, start_title, end_title
):
    text = ""
    header_height = header_px * 72 / 96  # 픽셀 단위를 72 dpi 단위로 변환
    footer_height = footer_px * 72 / 96  # 픽셀 단위를 72 dpi 단위로 변환
    for page_num in range(start_page, end_page + 1):
        page = pdf_document[page_num - 1]
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, _, _, _ = block
            if y0 > header_height and y1 < page.rect.height - footer_height:
                block_text = block[4].replace("\n", " ")

                text += (
                    block_text + " "
                )  # 줄바꿈을 공백으로 대체하고 블록 끝에 공백 추가

        try:
            if start_page == end_page:
                text = text.split(start_title)[1].split(end_title)[0]
            elif page_num == start_page:
                text = text.split(start_title)[1]
            elif page_num == end_page:
                text = text.split(end_title)[0]
        except IndexError:
            print(
                f"Splitting error at page {page_num} for titles '{start_title}' and '{end_title}'"
            )

    return text.strip()  # 마지막 공백 제거


def save_text_to_file(start_title, page_text, level, start_page, end_page):
    filename = f"C:/dev/ai/data_result/extracted_texts/{f"{start_page}-{end_page}"}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {start_title}\n")
        f.write(page_text)


def save_toc_and_contents(toc_texts):
    toc_filename = "C:/dev/ai/data_result/extracted_toc.txt"
    with open(toc_filename, "w", encoding="utf-8") as f:
        for title, text in toc_texts:
            f.write(f"{title}\n")

    contents_filename = "C:/dev/ai/data_result/extracted_contents.txt"
    with open(contents_filename, "w", encoding="utf-8") as f:
        for title, text in toc_texts:
            f.write(f"== {title} ==\n")
            f.write(text)
            f.write("\n\n")


def main(pdf_path, header_px, footer_px):
    toc = extract_toc(pdf_path)
    toc_texts = extract_text_by_toc(pdf_path, toc, header_px, footer_px)

    save_toc_and_contents(toc_texts)
    print("작업이 완료되었습니다.")


if __name__ == "__main__":
    pdf_path = "C:/dev/ai/data/scapp.pdf"  # PDF 파일 경로를 설정하세요
    header_px = 150  # 헤더 높이 (픽셀 단위)
    footer_px = 0  # 푸터 높이 (픽셀 단위)
    main(pdf_path, header_px, footer_px)
