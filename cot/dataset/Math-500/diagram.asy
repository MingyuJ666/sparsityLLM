// 设置输出格式和大小
settings.outformat = "pdf";
settings.render = 0;
size(8cm, 8cm);  // 设置画布大小

// 在代码开头添加直角标记函数
void rightangle(pair A, pair B, pair C, real size=0.5) {
    pair u = unit(B-A);
    pair v = unit(C-A);
    draw(A + size*u -- A + size*(u+v) -- A + size*v, red);
}

// 定义点坐标
pair F, E, D;
F = (0,0);
E = (0,7);
D = (sqrt(51),7);

// 绘制三角形
draw(D--E--F--cycle, linewidth(1pt));  // 绘制三角形，cycle 表示闭合

// 绘制直角标记（在点E处）
// draw(rightanglemark(D, E, F, 2.5));  // 大小为2.5
rightangle(E, F, D, 0.8);  // 在点E处绘制直角标记

// 添加点标签
label("$D$", D, NE);    // D点在右上方向
label("$E$", E, NW);    // E点在左上方向  
label("$F$", F, SW);    // F点在左下方向

// 添加边长标注
label("$7$", (E+F)/2, W);  // EF边中点左侧标注7

// 添加坐标轴（可选，帮助理解位置）
// xaxis(Arrow);
// yaxis(Arrow);

// 添加网格（可选，帮助理解位置）
// grid(10, 10, lightgray);

// 添加其他边的长度标注（可选）
label("$\sqrt{51}$", (D+E)/2, N);  // DE边中点上方标注√51
label("$10$", (D+F)/2, SE);        // FD边中点右下标注10