// Copied from hough_lines.py output
const jumpStats = [
    {
        jumpDist: 94.2,
        jumpMargin: -4.5,
        landMargin: 6.9,
    },
    {
        jumpDist: 99.5,
        jumpMargin: -1.1,
        landMargin: 4.9,
    },
    {
        jumpDist: 109.7,
        jumpMargin: 14.5,
        landMargin: 10.4,
    },
    {
        jumpDist: null,
        jumpMargin: null,
        landMargin: null,
    },
    {
        jumpDist: null,
        jumpMargin: null,
        landMargin: null,
    },
    {
        jumpDist: null,
        jumpMargin: null,
        landMargin: null,
    },
];

let selectedVideoId = 1;
let baseURL = 'https://www.hurdl.us';

const originalVid = document.getElementById('original-video');
const footPlacementVid = document.getElementById('foot-placement-vid');
const blobPlacementVid = document.getElementById('blob-placement-vid');
const loadCards = () => {
    const currentSelectedVideoId = selectedVideoId;
    let originalVideoPromise = new Promise((resolve, reject) => {
        originalVid.addEventListener('canplaythrough', () => resolve(() => originalVid.play()));
        originalVid.src = `${baseURL}/static/media/${selectedVideoId}/video_frames/video.mp4`;
    });

    let footPlacementVidPromise = new Promise((resolve, reject) => {
        footPlacementVid.addEventListener('canplaythrough', () => resolve(() => footPlacementVid.play()));
        footPlacementVid.src = `${baseURL}/static/media/${selectedVideoId}/foot_placement/video.mp4`;
    });

    let blobPlacementVidPromise = new Promise((resolve, reject) => {
        blobPlacementVid.addEventListener('canplaythrough', () => resolve(() => blobPlacementVid.play()));
        blobPlacementVid.src = `${baseURL}/static/media/${selectedVideoId}/blob_placement/video.mp4`;
    });

    Promise.all([originalVideoPromise, footPlacementVidPromise, blobPlacementVidPromise]).then(cbs => {
        cbs.forEach(cb => cb());
    });
}

const $vidSelect = $('#vid-select');
const $vidSelectTitle = $('#vid-select-title');
$('#vid-select a').click(function (e) {
    // remove active class
    $vidSelect.children().removeClass('active');
    // update selected video
    const $selectedOpt = $(e.currentTarget);
    selectedVideoId = parseInt($selectedOpt.text());
    $vidSelectTitle.text(`Video #${selectedVideoId}`)
    // add back active class
    $(e.currentTarget).addClass('active');
    loadCards();
});
loadCards()