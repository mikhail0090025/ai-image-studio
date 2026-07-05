const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const toolRemove = document.getElementById("toolRemove");
const toolEdit = document.getElementById("toolEdit");

const removeSettings = document.getElementById("removeSettings");
const editSettings = document.getElementById("editSettings");

const removeAutoDetect = document.getElementById("removeAutoDetect");
const editAutoDetect = document.getElementById("editAutoDetect");

const engine = document.getElementById("engine");
const steps = document.getElementById("steps");

const result = document.getElementById("result");

let mode = "remove";
let image = new Image();
let box = null;

// --------------------
// load image
// --------------------
const stored = sessionStorage.getItem("image");
if (stored) image.src = stored;

image.onload = () => {
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0);
};

// --------------------
// tool switching
// --------------------
toolRemove.onclick = () => {
  mode = "remove";

  toolRemove.classList.add("active");
  toolEdit.classList.remove("active");

  removeSettings.style.display = "block";
  editSettings.style.display = "none";
};

toolEdit.onclick = () => {
  mode = "edit";

  toolEdit.classList.add("active");
  toolRemove.classList.remove("active");

  removeSettings.style.display = "none";
  editSettings.style.display = "block";
};

// --------------------
// advanced
// --------------------
document.getElementById("toggleAdvanced").onclick = () => {
  const adv = document.getElementById("advanced");
  adv.style.display = adv.style.display === "none" ? "block" : "none";
};

// LLaMA = force 1 step
engine.onchange = () => {
  if (engine.value === "llama") {
    steps.value = 1;
    steps.disabled = true;
  } else {
    steps.disabled = false;
    steps.value = 15;
  }
};

// --------------------
// FIXED BOX SELECTION
// --------------------
let startX, startY, isDrawing = false;

function getCoords(e) {
  const rect = canvas.getBoundingClientRect();

  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

canvas.addEventListener("mousedown", (e) => {
  const p = getCoords(e);
  startX = p.x;
  startY = p.y;
  isDrawing = true;
});

canvas.addEventListener("mousemove", (e) => {
  if (!isDrawing) return;

  const p = getCoords(e);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);

  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;
  ctx.strokeRect(startX, startY, p.x - startX, p.y - startY);
});

canvas.addEventListener("mouseup", (e) => {
  const p = getCoords(e);
  isDrawing = false;

  box = [
    Math.round(startX),
    Math.round(startY),
    Math.round(p.x),
    Math.round(p.y)
  ];
});

// --------------------
// REMOVE
// --------------------
document.getElementById("runRemove").onclick = async () => {
  const form = new FormData();
  form.append("image", dataURLtoFile(image.src, "img.png"));

  const prompt = document.getElementById("removePrompt").value;

  if (!removeAutoDetect.checked && box) {
    form.append("box", JSON.stringify(box));
  }

  if (removeAutoDetect.checked && prompt) {
    form.append("prompt", prompt);
  }

  form.append("engine", engine.value);
  form.append("steps", engine.value === "llama" ? 1 : steps.value);

  const res = await fetch("/api/remove-object", {
    method: "POST",
    body: form
  });

  const blob = await res.blob();
  result.src = URL.createObjectURL(blob);
};

// --------------------
// EDIT
// --------------------
document.getElementById("runEdit").onclick = async () => {
  const form = new FormData();
  form.append("image", dataURLtoFile(image.src, "img.png"));

  const prompt = document.getElementById("editPrompt").value;

  if (!editAutoDetect.checked && box) {
    form.append("box", JSON.stringify(box));
  }

  form.append("edit_prompt", prompt);

  form.append("engine", engine.value);
  form.append("steps", engine.value === "llama" ? 1 : steps.value);

  const res = await fetch("/api/edit-object", {
    method: "POST",
    body: form
  });

  const blob = await res.blob();
  result.src = URL.createObjectURL(blob);
};

// --------------------
function dataURLtoFile(dataurl, filename) {
  const arr = dataurl.split(',');
  const mime = arr[0].match(/:(.*?);/)[1];
  const bstr = atob(arr[1]);
  let n = bstr.length;
  const u8arr = new Uint8Array(n);

  while (n--) u8arr[n] = bstr.charCodeAt(n);

  return new File([u8arr], filename, { type: mime });
}