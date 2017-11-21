$(document).ready(function() {
    $('#loader').hide();
    var inputs = document.querySelectorAll('.file-input');

    for (var i = 0, len = inputs.length; i < len; i++) {
        customInput(inputs[i])
    }

    function customInput(el) {
        const fileInput = el.querySelector('[type="file"]')
        const label = el.querySelector('[data-js-label]')

        fileInput.onchange =
            fileInput.onmouseout = function() {
                if (!fileInput.value) return

                var value = fileInput.value.replace(/^.*[\\\/]/, '')
                el.className += ' -chosen'
                label.innerText = value
            }
    }

    $( "#form" ).submit(function( event ) {
        $('#formDiv').hide();
        $('#loader').show();
    });


    $("#modelType")
        .change(function() {
            $("#modelType option:selected").each(function() {
                var mode = $(this).val();
                if (mode == 0) {
                    $('#modelUpload').hide();
                } else {
                    $('#modelUpload').show();
                }
            });
        });

    $("#vizType")
        .change(function() {
            $("#vizType option:selected").each(function() {
                var mode = $(this).val();
                if (mode == 1) {
                    $('#pictureUpload').show();
                } else {
                    $('#pictureUpload').hide();
                }
            });
        });
});
