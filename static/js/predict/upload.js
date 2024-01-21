var addButton = document.getElementById('addButton');
var textElementsDiv = document.getElementById('textElements');

addButton.addEventListener('click', function () {
    var textElementDiv = document.createElement('div');
    var input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Add your question here!';
    var deleteButton = document.createElement('button');
    deleteButton.classList.add('deleteButton');
    var icon = document.createElement("i");
    icon.className = "fas fa-trash";
    deleteButton.appendChild(icon);

    deleteButton.addEventListener('click', function () {
        textElementDiv.remove();
    });

    textElementDiv.appendChild(input);
    textElementDiv.appendChild(deleteButton);
    textElementsDiv.appendChild(textElementDiv);
});


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

const Modal = Swal.mixin({
    heightAuto: false
})

function upload_questions_and_pdf() {
    const textInputs = textElementsDiv.querySelectorAll('input[type="text"]');

    const pdfs = fileInput.files[0];
    const questions = Array.from(textInputs).map(input => input.value);

    if (!pdfs || questions.some(value => value === '')) {
        Modal.fire({ icon: "error", title: "Incomplete input!", text: "Please fill out all the question boxes and select a file." });
        return;
    }

    const formData = new FormData();
    formData.append('pdfs', pdfs);
    formData.append('questions', JSON.stringify(questions));

    Modal.fire({
        title: 'Please wait...',
        text: 'Your model is working.',
        allowOutsideClick: false,
        allowEscapeKey: false,
        allowEnterKey: false,
        showConfirmButton: false,
    });
    Modal.showLoading();

    fetch('/predict/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => {
            Modal.close();
            if (response.ok) {
                console.log('Success');
                window.location.href = '/predict/render';
            }
            else {
                console.log('Error');
                console.log(response);
            }
        });
}
