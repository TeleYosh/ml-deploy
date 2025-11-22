// home button for refresh
const homeBtn = document.querySelector('.home');
homeBtn.addEventListener('click', (e) => {
  window.location.reload();
})

// clear button
const clearBtn = document.querySelector('.clear');
clearBtn.addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  clear_result = {
    'prediction': [],
    'proba':[]
  };
  updateOutput(clear_result);
});

// prediction button
const predBtn = document.querySelector('.predict');
predBtn.addEventListener('click', async () => {
  // add white background
  ctx.globalCompositeOperation = 'destination-over';
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.globalCompositeOperation = 'source-over';
  const formData = new FormData();
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png'))
  formData.append('file', blob, 'drawing.png');
  try {
    const response = await fetch('http://127.0.0.1:8000/predict', {
    // const response = await fetch('/sketch/api/predict', {
      method: 'POST',
      body: formData
    });
    if (!response.ok) {
      throw new Error('Prediction request failed.');
    }
    const result = await response.json();
    updateOutput(result);
  } catch (err) {
    console.error(err);
    alert('Error during prediction.');
  }
});

// canvas
const canvas = document.getElementById('myCanvas');
const ctx = canvas.getContext('2d');
function resizeCanvas() {
  const sketchpad = document.getElementById('sketchpad');
  const sketchpadStyles = window.getComputedStyle(sketchpad);
  const sketchpadPadding = parseFloat(sketchpadStyles.paddingLeft) + parseFloat(sketchpadStyles.paddingRight);
  
  // Get actual sketchpad width and subtract padding
  const availableWidth = sketchpad.clientWidth - sketchpadPadding;
  const size = Math.min(availableWidth, 400);
  
  canvas.width = size;
  canvas.height = size;
}

resizeCanvas();
window.addEventListener('resize', resizeCanvas);

let isPainting = false;
let lineWidth = 15;
let startX;
let startY;

const getCoordinates = (e) => {
  const rect = canvas.getBoundingClientRect();
  if (e.touches && e.touches.length > 0) {
    return {
      x: e.touches[0].clientX - rect.left,
      y: e.touches[0].clientY - rect.top
    }
  }
  else {
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    }
  }
}

const draw = (e) => {
  if (!isPainting) {
    return;
  }
  e.preventDefault();
  const coords = getCoordinates(e)
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineTo(coords.x, coords.y);
  ctx.stroke();
}

const startDrawing = (e) => {
  isPainting = true;
  const coords = getCoordinates(e);
  startX = coords.x;
  startY = coords.y;
  ctx.beginPath();
  ctx.moveTo(startX, startY);
}

const stopDrawing = (e) => {
  if (!isPainting) return;
  isPainting = false;
  ctx.stroke();
  ctx.beginPath();
  predBtn.click();
}

// Mouse events
canvas.addEventListener('mousedown', startDrawing)
canvas.addEventListener('mouseup', stopDrawing)
canvas.addEventListener('mousemove', draw);
// Touch screen events
canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchend', stopDrawing);
canvas.addEventListener('touchmove', draw);


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

const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

if (!isTouchDevice){
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
};

// change output div
const output = document.querySelector('.output');
function updateOutput(result) {
  const output = document.querySelector('.output');
  const title = output.querySelector('.title');

  // Clear any old prediction items (keep the title/type headers)
  const oldItems = output.querySelectorAll('.item');
  oldItems.forEach((item, idx) => {
    if (idx > 0) item.remove(); // keep the first template item
  });

  const template = output.querySelector('.item');
  const labels = result.predictions || [];
  const probas = result.probas || [];

  if (labels.length === 0) {
    title.textContent = 'No drawing detected';
    template.querySelector('.label').textContent = 'No drawing detected';
    template.querySelector('.proba').textContent = '100%';
    template.querySelector('.bar').style.width = '100%';
    return;
  }

  // Update the first (template) item
  template.querySelector('.label').textContent = labels[0];
  template.querySelector('.proba').textContent = `${(probas[0] * 100).toFixed(1)}%`;
  template.querySelector('.bar').style.width = `${(probas[0] * 100).toFixed(1)}%`;

  // Clone and populate additional items for top 5
  for (let i = 1; i < Math.min(5, labels.length); i++) {
    const clone = template.cloneNode(true);
    clone.querySelector('.label').textContent = labels[i];
    clone.querySelector('.proba').textContent = `${(probas[i] * 100).toFixed(1)}%`;
    clone.querySelector('.bar').style.width = `${(probas[i] * 100).toFixed(1)}%`;
    output.appendChild(clone);
  }

  // Update title text
  title.textContent = `${labels[0]} (${(probas[0] * 100).toFixed(1)}%)`;
}

// current time for footer
const time = new Date();
const footer = document.querySelector('.copyright');
footer.textContent += time.toDateString();

