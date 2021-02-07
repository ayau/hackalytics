(function() {

    const input = document.getElementById('file-input');
    const video = document.getElementById('video');
    const videoSource = document.createElement('source');
    const stats = document.getElementById('stats');

    const handleStats = () => {
        fetch('http://localhost:5000/fetch_video_stats', {
            method: 'GET',
        }).then(response => {
            if (response.status === 200) {
                stats.innerText = 'Stats Calculated';
            } else {
                alert('Error calculating stats')
            }
        })
    }

    const uploadVideo = file => {
        let formData = new FormData()
        formData.append('file', file)
        fetch('http://localhost:5000/upload_video', {
            method: 'POST',
            body: formData
        }).then(response => {
            if (response.status === 200) {
                handleStats()
            } else {
                alert('Issue uploading')
            }
        })
    }

    input.addEventListener('change', function() {
        const files = this.files || [];

        if (!files.length) return;


        const reader = new FileReader();

        reader.onload = function (e) {
            stats.innerText = 'Loading Stats...';
            // remove old source if its there
            video.pause();
            video.removeAttribute('src');

            videoSource.setAttribute('src', e.target.result);
            video.appendChild(videoSource);
            video.load();
            video.play();

            uploadVideo(files[0]);
        };

        reader.readAsDataURL(files[0]);
    });
})();