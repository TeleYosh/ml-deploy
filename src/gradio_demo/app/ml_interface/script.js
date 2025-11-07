const fileInput = document.getElementById('avatar');
const windowDiv = document.querySelector('.window');
const uploadLabel = document.querySelector('.upload-label');

// Listen for file selection
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

const clearBtn = document.querySelector('.clear');

clearBtn.addEventListener('click', () => {
  const preview = document.querySelector('.preview-image');
  if (preview) preview.remove();
  uploadLabel.style.display = 'flex';
  fileInput.value = ''; // reset the file input
});

// current time
const time = new Date();
const footer = document.querySelector('.copyright');
footer.textContent += time.toDateString();

