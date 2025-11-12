const fileInput = document.getElementById('avatar');
const windowDiv = document.querySelector('.window');
const uploadLabel = document.querySelector('.upload-label');
const output = document.querySelector('.output');


// clear button
const clearBtn = document.querySelector('.clear');
clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
});

// prediction button
const predBtn = document.querySelector('.predict');
predBtn.addEventListener('click', async () => {
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
canvas.width = 300;
canvas.height = 300;

let isPainting = false;
let lineWidth = 25;
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
});
canvas.addEventListener('mousemove', draw);


// current time for footer
const time = new Date();
const footer = document.querySelector('.copyright');
footer.textContent += time.toDateString();

