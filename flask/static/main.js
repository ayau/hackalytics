// Copied from hough_lines.py output
const jumpStats = [
    {
        jumpDist: 94.2,
        jumpMargin: -4.5,
        landMargin: 6.9,
        speed:21.41,
    },
    {
        jumpDist: 99.5,
        jumpMargin: -1.1,
        landMargin: 4.9,
        speed:24.9,
    },
    {
        jumpDist: 109.7,
        jumpMargin: 14.5,
        landMargin: 10.4,
        speed:21.1,
    },
    {
        jumpDist: null,
        jumpMargin: null,
        landMargin: null,
        speed:null,
    },
    {
        jumpDist: null,
        jumpMargin: null,
        landMargin: null,
        speed:null,
    },
    {
        jumpDist: null,
        jumpMargin: null,
        landMargin: null,
        speed:null,
    },
];

let selectedVideoId = 1;
let baseURL = ''; //https://www.hurdl.us';

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

    // Update stats
    const stat = jumpStats[selectedVideoId - 1];
    $('#stat-jump-distance').text(stat.jumpDist ? stat.jumpDist + '"' : '-');
    $('#stat-speed').text(stat.speed ? stat.speed + ' ft/s' : '-');
    $('#stat-jump-margin').text(stat.jumpMargin ? (stat.jumpMargin > 0 ? stat.jumpMargin + '" before line' : (-1 * stat.jumpMargin) + '" after line') : '-');
    $('#stat-land-margin').text(stat.landMargin ? (stat.landMargin > 0 ? stat.landMargin + '" before line' : (-1 * stat.landMargin) + '" after line') : '-');

    Promise.all([originalVideoPromise, footPlacementVidPromise, blobPlacementVidPromise]).then(cbs => {
        cbs.forEach(cb => cb());
    });
}

const $vidSelect = $('#vid-select');
const $vidSelectTitle = $('#vid-select-title');
$('.thumb').click(function (e) {
    const vidId = $(this).data('id');
    if (selectedVideoId != vidId) {
        $('.thumb.active').removeClass('active');
        $(this).addClass('active');
        selectedVideoId = vidId;
        loadCards();
    }
});

loadCards()