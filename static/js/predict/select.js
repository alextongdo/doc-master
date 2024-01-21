var options = {
    valueNames: ['model_id', 'architecture', 'username', 'datetime']
};
var myList = new List('selectionTable', options);

const tbody = document.querySelector('tbody');
const observer = new MutationObserver(() => {
    updateSelectButtonState();
});
const config = { childList: true };
observer.observe(tbody, config);

const tableBody = document.getElementById('table-body');
const selectButton = document.getElementById('model-select');
tableBody.addEventListener('change', () => {
    updateSelectButtonState();
});

const Modal = Swal.mixin({
    heightAuto: false
})

function updateSelectButtonState() {
    if (document.querySelector('input[name="model_id"]:checked')) {
        selectButton.disabled = false;
    } else {
        selectButton.disabled = true;
    }
}

function selectModel() {
    const model_id = { model_id: document.querySelector('input[name="model_id"]:checked').value };
    console.log('model_id: ' + model_id.model_id);

    Modal.fire({
        title: 'Please wait...',
        text: 'Your model is loading.',
        allowOutsideClick: false,
        allowEscapeKey: false,
        allowEnterKey: false,
        showConfirmButton: false,
    });
    Modal.showLoading();

    fetch('/predict/select', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(model_id)
    })
        .then(response => {
            Modal.close();
            if (response.ok) {
                console.log("Predict model selection success");
                window.location.href = '/predict/upload';
            }
            else {
                console.log("Predict model selection failure");
                Modal.fire({ icon: "error", title: "Try again!", text: "Sorry, we could not select that model." });
            }
        });
}