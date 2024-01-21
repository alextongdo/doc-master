const Toast = Swal.mixin({
    toast: true,
    position: "top-end",
    showConfirmButton: false,
    timer: 1500,
    timerProgressBar: true,
});

function setFormMessage(formElement, type, message) {
    const messageElement = formElement.querySelector(".form__message");

    messageElement.textContent = message;
    messageElement.classList.remove("form__message--success", "form__message--error");
    messageElement.classList.add(`form__message--${type}`);
}

function setInputError(inputElement, message) {
    inputElement.classList.add("form__input--error");
    inputElement.parentElement.querySelector(".form__input-error-message").textContent = message;
}

function clearInputError(inputElement) {
    inputElement.classList.remove("form__input--error");
    inputElement.parentElement.querySelector(".form__input-error-message").textContent = "";
}

function validUserLength(username, length) {
    if (username.length < length) {
        return false;
    }
    return true;
}

function validPasswordPair(password, confirmPass) {
    if (password == "" || confirmPass == "") {
        return false;
    }
    if (password != confirmPass) {
        return false;
    }
    return true;
}

document.addEventListener("DOMContentLoaded", () => {
    const loginForm = document.querySelector("#login");
    const createAccountForm = document.querySelector("#createAccount");

    document.querySelector("#linkCreateAccount").addEventListener("click", e => {
        e.preventDefault();
        loginForm.classList.add("form--hidden");
        createAccountForm.classList.remove("form--hidden");
    });

    document.querySelector("#linkLogin").addEventListener("click", e => {
        e.preventDefault();
        loginForm.classList.remove("form--hidden");
        createAccountForm.classList.add("form--hidden");
    });

    const create_username = document.querySelector("#signupUsername");
    create_username.addEventListener("blur", e => {
        if (!validUserLength(e.target.value, 8)) {
            setInputError(create_username, "Username must be at least 8 characters in length!");
        }
    });
    create_username.addEventListener("input", e => {
        clearInputError(create_username);
    });

    // Login Request
    loginForm.addEventListener("submit", e => {
        e.preventDefault();

        const formData = new FormData(loginForm);
        console.log(Object.fromEntries(formData.entries()));

        if (formData.get("username") == "" || formData.get("password") == "") {
            setFormMessage(loginForm, "error", "Invalid username/password combination!");
            return;
        }

        console.log("Making login request.")
        formData.append('type', 'login');
        fetch('/login', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (response.ok) {
                    console.log("Login Sucess");
                    window.location.href = "/";
                }
                else {
                    console.log("Login failure");
                    setFormMessage(loginForm, "error", "Invalid username/password combination!");
                }
            });

    });

    // Create Account Request
    createAccountForm.addEventListener("submit", e => {
        e.preventDefault();

        const formData = new FormData(createAccountForm);
        console.log(Object.fromEntries(formData.entries()));

        if (!validUserLength(formData.get("username"), 8)) {
            return;
        }
        if (!validPasswordPair(formData.get("password"), formData.get("confirmPass"))) {
            return;
        }

        console.log("Create account request.")
        formData.append('type', 'register');
        fetch('/login', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (response.ok) {
                    console.log("Account created, redirect to login");
                    Toast.fire({
                        icon: "success",
                        title: "Account created!"
                    });
                    const linkLogin = document.querySelector("#linkLogin");
                    linkLogin.click();
                }
                else {
                    console.log("Account not created");
                    setFormMessage(createAccountForm, "error", "That account could not be created, try again!");
                }
            });
    });

});