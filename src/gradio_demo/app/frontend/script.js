const fileInput = document.getElementById('avatar');
const windowDiv = document.querySelector('.window');
const uploadLabel = document.querySelector('.upload-label');
const output = document.querySelector('.output');

// upload button
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Create an image element
  const img = document.createElement('img');
  img.src = URL.createObjectURL(file);
  img.alt = 'Uploaded preview';
  img.style.maxWidth = '100%';
  img.style.maxHeight = '100%';
  img.style.objectFit = 'contain';
  img.style.borderRadius = '5px';

  // Remove the upload label (so the preview replaces it)
  uploadLabel.style.display = 'none';

  // If an old preview exists, remove it before showing new one
  const oldPreview = windowDiv.querySelector('.preview-image');
  if (oldPreview) oldPreview.remove();

  // Add a class for easy removal later
  img.classList.add('preview-image');

  // Add image to the window
  windowDiv.appendChild(img);
});

// clear button
const clearBtn = document.querySelector('.clear');
clearBtn.addEventListener('click', () => {
  const preview = document.querySelector('.preview-image');
  if (preview) preview.remove();
  uploadLabel.style.display = 'flex';
  fileInput.value = ''; // reset the file input
});

// prediction button
const predBtn = document.querySelector('.predict');
predBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert('Please upload an image first.');
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

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
    // console.log(`tibari ya khoya ${result.proba}`)
  } catch (err) {
    console.error(err);
    alert('Error during prediction.');
  }
});

// current time for footer
const time = new Date();
const footer = document.querySelector('.copyright');
footer.textContent += time.toDateString();

