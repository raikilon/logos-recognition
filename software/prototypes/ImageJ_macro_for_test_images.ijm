function action(input, output, filename) {

open(input + filename);
run("Flip Horizontally");
saveAs("PNG", output + "h" + filename);
close();

open(input + filename);
run("Flip Vertically");
saveAs("PNG", output + "v" + filename);
close();

open(input + filename);
run("Rotate 90 Degrees Right");
saveAs("PNG", output + "90"+ filename);
close();

open(input + filename);
run("Rotate... ", "angle=50 grid=1 interpolation=Bilinear enlarge");
saveAs("PNG", output + "50"+ filename);
close();

open(input + filename);
run("8-bit");
saveAs("PNG", output +"8bit" +filename);
close();

open(input + filename);
run("8-bit");
run("Salt and Pepper");
saveAs("PNG", output +"salt" +filename);
close();

open(input + filename);
run("Gaussian Blur...", "sigma=2");
saveAs("PNG", output +"gauss" +filename);
close();

open(input + filename);
run("8-bit");
run("Add Noise");
saveAs("PNG", output +"noise" +filename);
close();

open(input + filename);
run("Scale...", "x=0.1 y=0.1 width=26 height=18 interpolation=Bilinear average create");
saveAs("PNG", output +"scale" +filename);
close();


open(input + filename);
run("Scale...", "x=- y=- width=79 height=150 interpolation=Bilinear");
saveAs("PNG", output +"scale"+ filename);
close();

}

input = "C:/Dev/git/Logos-Recognizion-for-Webshop-Services/logorec/resources/images/test/Visa/";
output = "C:/Users/nolti/Desktop/script/";

list = getFileList(input);
for (i = 0; i < list.length; i++)
        action(input, output, list[i]);