import re
from random import choice
import fitz

def get_word_index_from_char_index(text, char_index):
    char_intervals = [(m.start(), m.end()-1) for m in re.finditer(r'\S+', text)]
    for index, interval in enumerate(char_intervals):
        if interval[0] <= char_index <= interval[1]:
            return index
    return None

def get_bbox_from_char_index(text, char_index, word_info):
    word_index = get_word_index_from_char_index(text, char_index)
    if word_index != None:
        if len(word_info) < word_index:
            return (0,0,0,0,"")
        else:
            return word_info[word_index][:5]
    else:
        # Find the nearest bbox
        offset = 1
        nearest = None
        while nearest == None:
            nearest = get_word_index_from_char_index(text, char_index + offset)
            if nearest != None:
                break
            nearest = get_word_index_from_char_index(text, char_index - offset)
            offset += 1
        if len(word_info) >= nearest:
            return word_info[nearest][:5]
        else:
            return (0,0,0,0,"")

def get_answer_bbox(pred_json, context, bbox_list, k=1):
    top_k_bbox = {}
    for i in range(k):
        offset = pred_json[i]['offsets']
        pred_offsets = [m.start() + offset[0] for m in re.finditer(r'\S+', pred_json[i]['text'])]
        top_k_bbox['top' + str(i)] = [get_bbox_from_char_index(context, ind, bbox_list) for ind in pred_offsets]
    return top_k_bbox

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height))
    ]


def highlight(pdf_bytes, answer_bbox):

    colors = [(234, 153, 153),
              (249, 203, 156),
              (255, 229, 153),
              (182, 215, 168),
              (162, 196, 201),
              (164, 194, 244),
              (180, 167, 214),
              (213, 166, 189)]

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    answer_pages = answer_bbox

    for (question, top_k) in answer_bbox.items():
        rgb = choice(colors)
        for (k, bbox) in top_k.items():
            answer_pages[question][k] = {'text': "", 'page_num': -1, 'color': ()}
            if len(bbox) > 0:
                answer_pages[question][k]['color'] = rgb

                for bbox_word in bbox:
                    x1 = bbox_word[0]
                    y1 = bbox_word[1]
                    x2 = bbox_word[2]
                    y2 = bbox_word[3]
                    word = bbox_word[4]
                    rect = fitz.Rect(x1,y1,x2,y2)

                    for page_number in range(doc.page_count):
                        current_page = doc.load_page(page_number)
                        word_match = current_page.get_textbox(rect).strip()
                        print(f"word match for page {page_number} - {repr(word_match)}", flush=True)
                        if word in word_match or word_match in word:
                            answer_pages[question][k]['text'] += f'{word} '
                            if answer_pages[question][k]['page_num'] == -1:
                                answer_pages[question][k]['page_num'] = page_number
                            highlight = current_page.add_highlight_annot(quads=[rect])
                            stroke = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
                            highlight.set_colors(stroke=stroke)
                            highlight.update()
                            break

                answer_pages[question][k]['text'] = answer_pages[question][k]['text'].rstrip()

    doc_bytes = doc.tobytes()
    doc.close()
    return doc_bytes, answer_pages

