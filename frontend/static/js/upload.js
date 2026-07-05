const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const selectBtn = document.getElementById("selectBtn");
const preview = document.getElementById("preview");
const continueBtn = document.getElementById("continueBtn");

let selectedFile = null;

selectBtn.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
  handleFile(e.target.files[0]);
};

dropZone.ondragover = (e) => {
  e.preventDefault();
};

dropZone.ondrop = (e) => {
  e.preventDefault();
  handleFile(e.dataTransfer.files[0]);
};

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) return;

  selectedFile = file;

  const url = URL.createObjectURL(file);
  preview.src = url;

  preview.classList.remove("hidden");
  dropZone.querySelector(".drop-content").style.display = "none";

  continueBtn.disabled = false;
}

continueBtn.onclick = () => {
  const reader = new FileReader();

  reader.onload = () => {
    sessionStorage.setItem("image", reader.result);
    window.location.href = "/editor";
  };

  reader.readAsDataURL(selectedFile);
};