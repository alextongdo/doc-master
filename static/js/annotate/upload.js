const Modal = Swal.mixin({
    heightAuto: false
})

async function upload_pdf() {
    const pdfs = fileInput.files[0];

    if (!pdfs) {
        Modal.fire({ icon: "error", title: "No file!", text: "Please upload a PDF or ZIP." });
        return;
    }

    const formData = new FormData();
    formData.append('pdfs', pdfs);

    let response = await fetch('/annotate/upload', {
        method: 'POST',
        body: formData
    });
    let data = await response.json();

    if (response.ok) {
        console.log('Success');
        console.log(data);
        if (data.success.length > 0) {
            Modal.fire({ icon: "warning", title: "Error!", text: data.success });

            Modal.fire({
                title: "Error!",
                text: data.success,
                icon: "warning",
            }).then((result) => {
                if (result.isConfirmed || result.isDismissed) {
                    // Redirect the user
                    window.location.href = '/annotate/render';
                }
            });
        }
        else {
            window.location.href = '/annotate/render';
        }
    }
    else {
        console.log('Error');
        console.log(data);
        Modal.fire({ icon: "error", title: "Error!", text: data.error });
    }
}

const fileInput = document.getElementById('fileInput');
const fileText = document.querySelector('#fileNameLabel span');

fileInput.addEventListener('change', function () {
    const files = fileInput.files;
    if (files.length > 0) {
        fileText.innerText = files[0].name;
    } else {
        fileText.innerText = 'PDF or ZIP';
    }
});