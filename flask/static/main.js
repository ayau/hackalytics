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