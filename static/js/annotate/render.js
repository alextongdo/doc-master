var pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';

function renderTextLayer(page, viewport, div) {
    return page.getTextContent().then((textContent) => {
        return pdfjsLib.renderTextLayer({
            textContentSource: textContent,
            container: div,
            viewport: viewport,
            enhanceTextSelection: true,
        }).promise;
    });
}

class ExtendedPDFjsViewer extends PDFjsViewer {
    _renderPage(page, i) {
        let pageinfo = this.pages[i];
        let pixel_ratio = Math.max(window.devicePixelRatio || 1, 1);

        let viewport = page.getViewport({
            rotation: this._rotation,
            scale: this._zoom.current * pixel_ratio
        });
        pageinfo.width = viewport.width / this._zoom.current / pixel_ratio;
        pageinfo.height = viewport.height / this._zoom.current / pixel_ratio;
        pageinfo.$div.data("width", pageinfo.width);
        pageinfo.$div.data("height", pageinfo.height);
        pageinfo.loaded = true;

        let $canvas = $("<canvas>");
        let canvas = $canvas.get(0);
        canvas.height = viewport.height;
        canvas.width = viewport.width;

        let ctx = canvas.getContext('2d');
        let renderTask = page.render({
            canvasContext: ctx,
            viewport: viewport,
        }).promise;

        let $textLayerDiv = $("<div class='textLayer'>");
        let textLayerTask = renderTextLayer(page, viewport, $textLayerDiv.get(0));

        $textLayerDiv.on('mouseup keyup', function (event) {
            console.log("textlayer selection");
            addSelectionToBox();
        });

        return Promise.all([renderTask, textLayerTask]).then(() => {

            let scale_factor = (pageinfo.$div.width() / pageinfo.width)
            $textLayerDiv.css("--scale-factor", scale_factor);
            $textLayerDiv.attr("tabindex", -1);

            $textLayerDiv.find('span').each(function () {
                const regExp = /(\d+(\.\d+)?)(?=px)/g;
                let font_multiplier = $(this).get(0).style.fontSize.match(regExp);
                let decimal_font_size = scale_factor * font_multiplier;
                let rounded_font_size = Math.round(decimal_font_size);
                let font_scale = decimal_font_size / rounded_font_size;

                let scaleX = $(this).get(0).style.transform
                let new_transform = scaleX + ` scale(${font_scale})`;
                $(this).css('transform', new_transform);
            });

            $textLayerDiv.find('span').filter(function () {
                return $(this).text().trim() === '';
            }).remove();

            let $contentWrapper = $("<div class='contentWrapper'>").append($canvas, $textLayerDiv);
            this._setPageContent(pageinfo.$div, $contentWrapper);

            if (typeof this.settings.onPageRender === "function") {
                this.settings.onPageRender.call(this, pageinfo.$div, i);
            }
            return pageinfo;
        });
    }
}

function resetQuestionAnswers() {
    const textElementsDiv = document.querySelector("#textElements");
    while (textElementsDiv.lastElementChild) {
        textElementsDiv.removeChild(textElementsDiv.lastElementChild);
    }
}

function addQuestion(metadata) {
    let textElementsDiv = document.getElementById('textElements');
    let textElementDiv = document.createElement('div');

    let inputsDiv = document.createElement('div');
    let input = document.createElement('textarea');
    input.placeholder = 'Add your question here!';
    input.rows = 1;

    let highlight = document.createElement('textarea');
    highlight.readOnly = true;
    highlight.placeholder = 'Highlight an answer in the PDF!';
    highlight.rows = 2;

    if (metadata !== undefined) {
        let annot_data = metadata;
        if (typeof annot_data["anchor"] == "string") {
            annot_data["anchor"] = JSON.parse(annot_data["anchor"]);
        }
        if (typeof annot_data["focus"] == "string") {
            annot_data["focus"] = JSON.parse(annot_data["focus"]);
        }
        console.log(annot_data);
        input.value = annot_data["question"];
        highlight.value = annot_data["answer"];
        highlight.dataset.anchor = JSON.stringify(annot_data["anchor"]);
        highlight.dataset.focus = JSON.stringify(annot_data["focus"]);
        highlight.dataset.page_size = annot_data["page_size"];
    }

    inputsDiv.appendChild(input);
    inputsDiv.appendChild(highlight);

    let deleteButton = document.createElement('button');
    deleteButton.classList.add('deleteButton');
    let icon = document.createElement("i");
    icon.className = "fas fa-trash";
    deleteButton.appendChild(icon);

    deleteButton.addEventListener('click', function () {
        textElementDiv.remove();
    });

    textElementDiv.appendChild(inputsDiv);
    textElementDiv.appendChild(deleteButton);
    textElementsDiv.appendChild(textElementDiv);
}

function addSelectionToBox() {
    let focus_box = document.querySelector('.focus');
    if (focus_box == null) return;

    let highlight_box = focus_box.children[1];
    if (window.getSelection) {
        let selection = window.getSelection();
        highlight_box.value = selection.toString();

        const anchor_span = selection.anchorNode.parentElement;
        const anchor_span_style = window.getComputedStyle(anchor_span);
        const anchor_left = parseFloat(anchor_span_style.getPropertyValue("left"));
        const anchor_top = parseFloat(anchor_span_style.getPropertyValue("top"));
        const anchor = {
            "text": selection.anchorNode.textContent,
            "page_num": anchor_span.parentElement.parentElement.parentElement.parentElement.dataset.page,
            "bbox": [
                anchor_left,
                anchor_top,
                anchor_left + parseFloat(anchor_span.getBoundingClientRect().width),
                anchor_top + parseFloat(anchor_span.getBoundingClientRect().height)
            ],
            "offset": selection.anchorOffset,
        };

        console.log("anchor");
        console.log(anchor);
        console.log(selection.anchorOffset);
        highlight_box.dataset.anchor = JSON.stringify(anchor);


        const focus_span = selection.focusNode.parentElement;
        const focus_span_style = window.getComputedStyle(focus_span);
        const focus_left = parseFloat(focus_span_style.getPropertyValue("left"));
        const focus_top = parseFloat(focus_span_style.getPropertyValue("top"));

        const focus = {
            "text": selection.focusNode.textContent,
            "page_num": focus_span.parentElement.parentElement.parentElement.parentElement.dataset.page,
            "bbox": [
                focus_left,
                focus_top,
                focus_left + parseFloat(focus_span.getBoundingClientRect().width),
                focus_top + parseFloat(focus_span.getBoundingClientRect().height)
            ],
            "offset": selection.focusOffset,
        };
        console.log("focus");
        console.log(focus);
        console.log(selection.focusOffset);
        highlight_box.dataset.focus = JSON.stringify(focus);

        const textlayer = anchor_span.parentElement;
        const textlayer_style = window.getComputedStyle(textlayer);
        const page_size = [
            parseFloat(textlayer_style.getPropertyValue("width")),
            parseFloat(textlayer_style.getPropertyValue("height"))
        ];
        console.log("pagesize");
        console.log(page_size);
        highlight_box.dataset.page_size = JSON.stringify(page_size);
    }
}

function download() {
    fetch("/annotate/get_all", { method: "GET" })
        .then(response => response.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const downloadLink = document.createElement("a");
            downloadLink.href = url;
            downloadLink.download = "annotations.csv";
            downloadLink.click();
            URL.revokeObjectURL(url);
        })
        .catch(error => {
            console.error("Error downloading CSV:", error);
        });
}

const Modal = Swal.mixin({
    heightAuto: false
});
const Toast = Swal.mixin({
    toast: true,
    position: "top-end",
    showConfirmButton: false,
    timer: 1500,
    timerProgressBar: true,
});
var curr_pdf_index = 0;
const pdfViewer = new ExtendedPDFjsViewer($(".pdfjs-viewer"));

async function saveAnnotations() {
    try {
        let response = await fetch(`/annotate/${index}`, {
            method: "POST",
        });
        if (response.ok) {
            return true;
        }
        else {
            return false;
        }
    } catch (error) {
        console.log(error);
        return false;
    }
}
async function getPDF(index) {
    try {
        let response = await fetch(`/annotate/${index}`);
        if (response.ok) {
            let data = await response.json();
            const pdfArray = Uint8Array.from(atob(data.pdf_bytes), (c) => c.charCodeAt(0));

            const filename_element = document.querySelector("#fileName p");
            filename_element.textContent = data.filename;

            resetQuestionAnswers();
            console.log(data);
            if (data.metadata.length === 0) {
                addQuestion();
            }

            for (let i = 0; i < data.metadata.length; i++) {
                addQuestion(data.metadata[i]);
            }

            await pdfViewer.loadDocument(pdfArray);
            pdfViewer.setZoom("width");

            return true;
        } else {
            console.log(response);
            return false;
        }
    } catch (error) {
        console.log(error);
        return false;
    }
}

async function nextPDF() {
    const textareaDivs = document.querySelectorAll("#textElements>div>div");
    if (textareaDivs.length == 0) {
        Modal.fire({ icon: "error", title: "Annotation incomplete!", text: "Please add some questions and answers." });
        return;
    }

    const annotations = [];

    for (const input of textareaDivs) {
        const annotation = {};
        annotation["question"] = input.children[0].value.trim();
        annotation["answer"] = input.children[1].value.trim();
        if (annotation["question"].length == 0 || annotation["answer"].length == 0) {
            Modal.fire({ icon: "error", title: "Annotation incomplete!", text: "Please fill out all questions and answers." });
            return;
        }
        annotation["anchor"] = input.children[1].dataset.anchor;
        annotation["focus"] = input.children[1].dataset.focus
        annotation["page_size"] = input.children[1].dataset.page_size;
        annotations.push(annotation);
    }

    console.log(annotations);

    const formData = new FormData();
    formData.append('annotations', JSON.stringify(annotations));

    let response = await fetch(`/annotate/${curr_pdf_index}`, {
        method: "POST",
        body: formData
    });
    if (response.ok) {
        const success = await getPDF(curr_pdf_index + 1);
        if (success) {
            Toast.fire({
                icon: "success",
                title: "Annotations auto-saved!"
            });
            curr_pdf_index++;
        }
        else {
            Modal.fire({
                title: "Annotations finished!",
                text: "You will be redirected in a few seconds.",
                icon: "success",
                showConfirmButton: false,
                timer: 3000, // Adjust the timer as needed
            }).then((result) => {
                if (result.isDismissed) {
                    // Redirect the user
                    window.location.href = '/';
                }
            });
        }
    }
    else {
        Modal.fire({ icon: "error", title: "Save failed!", text: "Those annotations couldn't be saved." });
    }
}

async function prevPDF() {
    const success = await getPDF(curr_pdf_index - 1);
    if (success) {
        curr_pdf_index--;
    }
    else {
        Modal.fire({ icon: "error", title: "Couldn't get PDF!", text: "That was the first PDF." });
    }
}

document.addEventListener("click", function (event) {
    let focus_box = document.querySelector('.focus');
    let clicked = event.target;
    if (clicked == null) {
        if (focus_box != null) focus_box.classList.remove('focus');
    }
    else if (clicked.tagName == 'TEXTAREA') {
        if (focus_box != null) focus_box.classList.remove('focus');
        clicked.parentElement.classList.add('focus');
    }
    else if (!clicked.classList.contains('textLayer') && clicked.tagName != "SPAN") {
        if (focus_box != null) focus_box.classList.remove('focus');
    }
});

getPDF(curr_pdf_index);


