var options = {
    valueNames: ['username', 'datetime', 'num_docs']
};
var myList = new List('selectionTable', options);

const selectAllCheckbox = document.getElementById('selectAll');
const checkboxes = document.querySelectorAll('#selectionTable tbody input[type="checkbox"]');

selectAllCheckbox.addEventListener('change', function () {
    checkboxes.forEach(function (checkbox) {
        checkbox.checked = selectAllCheckbox.checked;
    });
});
checkboxes.forEach(function (checkbox) {
    checkbox.addEventListener('change', function () {
        // Check if all other checkboxes are checked
        var allChecked = Array.from(checkboxes).every(function (checkbox) {
            return checkbox.checked;
        });
        // Mark "select all" checkbox accordingly
        selectAllCheckbox.checked = allChecked;
    });
});

const Modal = Swal.mixin({
    heightAuto: false
})

function trainModel() {
    // Validate model selection
    const dropdown = document.getElementById("model-dropdown");
    const modelChoice = dropdown.value;
    if (modelChoice === "") {
        Modal.fire({ icon: "error", text: "Please choose a model architecture!" });
        return;
    }

    const selectedSessions = [];
    const selectedBoxes = document.querySelectorAll('#selectionTable tbody input[type="checkbox"]:checked');
    selectedBoxes.forEach((selection) => {
        const session_id = selection.dataset.session;
        selectedSessions.push(session_id);
    });
    if (!selectedSessions.length) {
        Modal.fire({ icon: "error", text: "Please select some annotations!" });
        return;
    }

    const modelName = document.getElementById("modelName");
    if (modelName.value === "") {
        Modal.fire({ icon: "error", text: "Please choose a name for your model!" });
        return;
    }
    if (existing_model_ids.includes(modelName.value)) {
        Modal.fire({ icon: "error", text: "Sorry, that model name is already being used. Please chose another one!" });
        return;
    }

    const data = {
        session_ids: selectedSessions,
        model_id: modelName.value,
        architecture: modelChoice
    };

    console.log(data);

    Modal.fire({
        title: 'Training',
        text: 'Please wait...',
        allowOutsideClick: false,
        allowEscapeKey: false,
        allowEnterKey: false,
        showConfirmButton: false,
    });
    Modal.showLoading();

    fetch('/train', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => {
            Modal.close();
            if (response.ok) {
                console.log(response);

                Modal.fire({
                    title: "Training in progress!",
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
            else {
                console.log("Training Failure");
                Modal.fire({ icon: "error", title: "Training failure!", text: "Please try again." });
            }
        });
}
