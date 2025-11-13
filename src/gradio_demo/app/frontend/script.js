// const fileInput = document.getElementById('avatar');
// const windowDiv = document.querySelector('.window');
// const uploadLabel = document.querySelector('.upload-label');
const output = document.querySelector('.output');


// clear button
const clearBtn = document.querySelector('.clear');
clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// prediction button
const predBtn = document.querySelector('.predict');
predBtn.addEventListener('click', async () => {
  ctx.globalCompositeOperation = 'destination-over';
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = 'source-over';
  const formData = new FormData();
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  formData.append('file', blob, 'drawing.png');
  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error('Prediction request failed.');
    }
    const result = await response.json();
    const labelEl = output.querySelector('.label');
    const probaEl = output.querySelector('.proba');
    labelEl.textContent = result.prediction;
    probaEl.textContent = `${(result.proba * 100).toFixed(1)}%`;
  } catch (err) {
    console.error(err);
    alert('Error during prediction.');
  }
});

// canvas
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
const canvasOffSetX = canvas.offsetLeft;
const canvasOffSetY = canvas.offsetTop;
console.log(`offset x ${canvasOffSetX} offset y ${canvasOffSetY}`)
canvas.width = 400;
canvas.height = 400;

let isPainting = false;
let lineWidth = 15;
let startX;
let startY;

const draw = (e) => {
  if (!isPainting) {
    return;
  }
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineTo(e.clientX-canvasOffSetX, e.clientY-canvasOffSetY);
  ctx.stroke();
}
canvas.addEventListener('mousedown', (e) => {
  isPainting = true;
  startX = e.clientX;
  startY = e.clientY;
});
canvas.addEventListener('mouseup', (e) => {
  isPainting = false;
  ctx.stroke();
  ctx.beginPath();
  predBtn.click();
});
canvas.addEventListener('mousemove', draw);

// --- Pen preview ---
const penPreview = document.createElement('div');
penPreview.style.position = 'absolute';
penPreview.style.width = `${lineWidth}px`;
penPreview.style.height = `${lineWidth}px`;
penPreview.style.border = '2px solid gray';
penPreview.style.borderRadius = '50%';
penPreview.style.pointerEvents = 'none'; // let mouse pass through
penPreview.style.display = 'none';
penPreview.style.zIndex = '1000';
document.body.appendChild(penPreview);

canvas.addEventListener('mouseenter', () => {
  penPreview.style.display = 'block';
});
canvas.addEventListener('mouseleave', () => {
  penPreview.style.display = 'none';
});
canvas.addEventListener('mousemove', (e) => {
  const x = e.clientX;
  const y = e.clientY;
  penPreview.style.left = `${x - lineWidth / 2}px`;
  penPreview.style.top = `${y - lineWidth / 2}px`;
});



// current time for footer
const time = new Date();
const footer = document.querySelector('.copyright');
footer.textContent += time.toDateString();

