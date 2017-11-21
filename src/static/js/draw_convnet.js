function drawConvnet(layer_data){
    var minX1 = minX2 = minX3 = 10000;

    for (i = 0; i < layer_data.length; i++) {
        minX1 = (layer_data[i].shape[2] < minX1) ? layer_data[i].shape[2] : minX1;
        minX2 = (layer_data[i].shape[1] < minX2 && layer_data[i].shape[1] != 1) ? layer_data[i].shape[1] : minX2;
        minX3 = (layer_data[i].shape[0] < minX3) ? layer_data[i].shape[0] : minX3;
    }

    var x = 50;
    for (i = 0; i < layer_data.length; i++) {
        x1 = layer_data[i].shape[2] * 7 / minX1;
        x1 = (x1 <= 49) ? x1 : 49;
        x2 = layer_data[i].shape[1] * 20 / minX2;
        x2 = (x2 <= 180) ? x2 : 180;
        x2 = (x2 >= 20) ? x2 : 20;
        var shiftX1 = x1 + x2 + 100;
        shiftX1 = (shiftX1 >= 100) ? shiftX1 : 100;
        x = x + shiftX1;
    }

    var canvas = document.getElementById('canvas');
    canvas.width = x;
    canvas.height = window.innerHeight / 2;

    var ctx = canvas.getContext('2d');

    var x = 50;

    var colors = ['#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f',
        '#cab2d6', '#ffff99', '#1f78b4', '#33a02c'
    ];

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (i = 0; i < layer_data.length; i++) {
            x1 = layer_data[i].shape[2] * 7 / minX1;
            x1 = (x1 <= 49) ? x1 : 49;
            x2 = layer_data[i].shape[1] * 20 / minX2;
            x2 = (x2 <= 180) ? x2 : 180;
            x2 = (x2 >= 20) ? x2 : 20;
            x3 = layer_data[i].shape[0] * 20 / minX3;
            x3 = (x3 <= 180) ? x3 : 180;

            drawCube(
                x, window.innerHeight / 4 + x3 / 2,
                x1, x2, x3,
                colors[i % colors.length],
                ctx
            );

            if (i != layer_data.length - 1)
                drawArrow(x + x2 + 45, window.innerHeight / 4 - 15, 10, colors[i % colors.length], ctx);


            ctx.font = "16px Raleway";
            ctx.fillStyle = "#333";
            ctx.textAlign = "center";
            ctx.fillText(layer_data[i].name, x, canvas.height - 100);
            ctx.fillText(layer_data[i].shape, x, canvas.height - 80);

            var shiftX1 = x1 + x2 + 100;
            shiftX1 = (shiftX1 >= 100) ? shiftX1 : 100;
            x = x + shiftX1;
        }
    }
    draw();

    var dragging = false;
    var lastX;
    var marginLeft = 0;

    canvas.addEventListener('mousedown', function(e) {
        var evt = e || event;
        dragging = true;
        lastX = evt.clientX;
        e.preventDefault();
    }, false);

    window.addEventListener('mousemove', function(e) {
        var evt = e || event;
        if (dragging) {
            var delta = evt.clientX - lastX;
            lastX = evt.clientX;
            marginLeft += delta;
            canvas.style.marginLeft = marginLeft + "px";
        }
        e.preventDefault();
    }, false);

    window.addEventListener('mouseup', function() {
        dragging = false;
    }, false);
}
