var pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.4.120/pdf.worker.min.js';

// Saves pdf bytes to download
let download_current_pdf = null;
let curr_pdf_index = 0;
const pdfViewer = new PDFjsViewer($(".pdfjs-viewer"));

const Modal = Swal.mixin({
    heightAuto: false
})

async function getPDF(index) {
    try {
        let response = await fetch(`/predict/${index}`);
        if (response.ok) {
            let data = await response.json();
            const pdfArray = Uint8Array.from(atob(data.pdf_bytes), (c) => c.charCodeAt(0));
            const filename = data.filename;

            const display_filename = document.querySelector("#fileName p");
            display_filename.textContent = filename;

            const answers = data.answers;
            updatePredictionTable(answers);

            download_current_pdf = data.pdf_bytes;

            await pdfViewer.loadDocument(pdfArray);
            pdfViewer.setZoom("width");

            return true;
        } else {
            console.log(response);
            return false;
        }
    } catch(error) {
        console.log(error);
        return false;
    }
}

function updatePredictionTable(prediction) {
    const table_body = document.querySelector("tbody");

    // Clear the table body, previous contents are removed before adding new rows
    while (table_body.lastElementChild) {
        table_body.removeChild(table_body.lastElementChild);
    }

    const template = document.getElementById("row-template");
    for (const question in prediction) {
        console.log("question is " + question);
        const row = template.content.cloneNode(true);
        const cols = row.querySelectorAll('td');
        const row_tr = row.querySelector('tr');
        const color = prediction[question]["top0"]["color"];

        if (color.length > 0) {
            cols[0].style.setProperty('--highlight-color', `rgb(${color[0]}, ${color[1]}, ${color[2]})`);
            row_tr.style.setProperty('--highlight-color', `rgb(${color[0]}, ${color[1]}, ${color[2]})`);
            row_tr.style.cursor = "pointer";
        }

        cols[0].textContent = question;

        const ans = prediction[question]["top0"]["text"];
        cols[1].textContent = (ans.length == 0) ? "N/A" : ans

        row_tr.addEventListener("click", function() {
            const go_to_page = prediction[question]["top0"]["page_num"];
            console.log(go_to_page);
            if (go_to_page >= 0) {
                pdfViewer.scrollToPage(go_to_page + 1);
            }
        });

        table_body.appendChild(row);
    }
}

async function nextPDF() {
    const success = await getPDF(curr_pdf_index + 1);
    if (success) {
        curr_pdf_index++;
    }
    else {
        Modal.fire({ icon: "error", title: "Couldn't get PDF!", text: "That was the last PDF." });
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

function download_current() {
    const decodedPdfData = atob(download_current_pdf);
    // Convert the decoded data to a Uint8Array
    const pdfByteArray = new Uint8Array(decodedPdfData.length);
    for (let i = 0; i < decodedPdfData.length; i++) {
      pdfByteArray[i] = decodedPdfData.charCodeAt(i);
    }
    const pdfBlob = new Blob([pdfByteArray], { type: 'application/pdf' });
    const pdfUrl = URL.createObjectURL(pdfBlob);
    const downloadLink = document.createElement('a');
    downloadLink.setAttribute("type", "hidden");
    downloadLink.href = pdfUrl;
    const display_filename = document.querySelector("#fileName p");
    const split = display_filename.textContent.split('.');
    downloadLink.download = split[0] + "_highlighted." + split[1]
    document.body.appendChild(downloadLink);
    downloadLink.click();
    URL.revokeObjectURL(pdfUrl);
    document.body.removeChild(downloadLink)
}

function download_all() {
    console.log("download all")
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '/predict/download_all', true);
    xhr.responseType = 'blob';
    xhr.onload = function(e) {
        if (this.status === 200) {
            const blob = new Blob([this.response], { type: 'application/zip' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'all_pdfs.zip';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };
    xhr.send();
};

getPDF(curr_pdf_index);