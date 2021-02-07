const framesRangeByVideo = [
    null, // ivan indexed with 1
    [55, 195],
    [48, 120],
    [219, 330],
    [3, 150],
    [8, 110],
    [103,279]
];

let selectedVideoId = 1;

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
});

let frameNumber = framesRangeByVideo[selectedVideoId][0];
const getNextFrameNumber = (prevSelectedVideoId) => {
    if (prevSelectedVideoId !== selectedVideoId) {
        frameNumber = framesRangeByVideo[selectedVideoId][0];
        return
    }

    frameNumber++;
    if (frameNumber > framesRangeByVideo[selectedVideoId][1]) {
        frameNumber = framesRangeByVideo[selectedVideoId][0];
    }
}

const sleep = (time) => {
    return new Promise((resolve) => setTimeout(resolve, time));
}

const cardImageSize = 360;
const $originalVidContainer = $('#original-vid-container');
const $footPlacementcontainer = $('#foot-placement-vid-container');
const $blobPlacementcontainer = $('#blob-placement-vid-container');
const loadCards = () => {
    const currentSelectedVideoId = selectedVideoId;
    let originalVideoPromise = new Promise((resolve, reject) => {
        const img = new Image(cardImageSize);
        img.addEventListener('load', () => resolve(() => $originalVidContainer.html(img)));
        img.addEventListener('error', (err) => resolve(() => {}));
        img.src = `/static/media/${selectedVideoId}/video_frames/${frameNumber}.jpg`;
    });

    let footPlacementPromise = new Promise((resolve, reject) => {
        const img = new Image(cardImageSize);
        img.addEventListener('load', () => resolve(() => $footPlacementcontainer.html(img)));
        img.addEventListener('error', (err) => resolve(() => {}));
        img.src = `/static/media/${selectedVideoId}/foot_placement/${frameNumber}.png`;
    });

    let blobPlacementPromise = new Promise((resolve, reject) => {
        const img = new Image(cardImageSize);
        img.addEventListener('load', () => resolve(() => $blobPlacementcontainer.html(img)));
        img.addEventListener('error', (err) => resolve(() => {}));
        img.src = `/static/media/${selectedVideoId}/blob_placement/${frameNumber}.png`;
    });

    sleep(40).then(() => {
        Promise.all([originalVideoPromise, footPlacementPromise, blobPlacementPromise]).then(cbs => {
            cbs.forEach(cb => cb());
            getNextFrameNumber(currentSelectedVideoId);
            loadCards()
        });
    });
}

loadCards(frameNumber)
