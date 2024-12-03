const form = document.getElementById("imageUploadForm");

const dropArea = document.querySelector(".drag-area");
const dragText = document.querySelector(".header");

let button = dropArea.querySelector(".button");
let input = dropArea.querySelector("input");
let submitBtn = document.querySelector("#submit");

let predictedConfidence = document.querySelector("#confidence");
let predictedLabel = document.querySelector("#label");
let predictedClass = document.querySelector("#predictedClass");

let file;

let preview = document.createElement("img");
preview.style.maxWidth = "20%";

// Function to enable the submit button
function enableSubmitButton() {
  submitBtn.disabled = false;
}

// Click button to open file dialog
button.onclick = () => {
  input.click();
};

input.addEventListener("change", function () {
  file = this.files[0];
  dropArea.classList.add("active");
  showPreview(file);
  enableSubmitButton();
});

// when file is inside drag area
dropArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropArea.classList.add("active");
  dragText.textContent = "Release to Upload";
});

// when file leave the drag area
dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("active");
  dragText.textContent = "Choose the file";
});

// when file is dropped
dropArea.addEventListener("drop", (event) => {
  event.preventDefault();
  file = event.dataTransfer.files[0]; // grab single file even of user selects multiple files
  showPreview(file);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  dropArea.classList.remove("active");
  dragText.textContent = "Choose the file";
  uploadFile(file);
});

function showPreview(file) {
  var reader = new FileReader();
  reader.onload = function (e) {
    preview.src = e.target.result;
    dropArea.appendChild(preview);
  };
  reader.readAsDataURL(file);
}

function uploadFile(file) {
  var formData = new FormData();
  formData.append("file", file);

  // Example POST request with FormData
  fetch("/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      predictedConfidence.textContent = `Confidence: ${data.confidence}`;
      predictedLabel.textContent = `Label: ${data.label}`;
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}