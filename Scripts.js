const imageInput = document.getElementById('imageInput');
        const imagePreview = document.getElementById('imagePreview');
        const movieResults = document.getElementById('movieResults');
        const loading = document.getElementById('loading');

        imageInput.addEventListener('change', function() {
            const file = imageInput.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Image Preview">`;
            };
            reader.readAsDataURL(file);
        });

        async function submitImage() {
            const file = imageInput.files[0];
            if (!file) {
                alert('Please select an image first!');
                return;
            }

            loading.style.display = 'block'; 
            movieResults.innerHTML = ''; 

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('http://127.0.0.1:8000/predict', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error('Error uploading image');
                }

                const resultHtml = await response.text();
                movieResults.innerHTML = resultHtml;
            } catch (error) {
                alert('Failed to submit the image: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        }