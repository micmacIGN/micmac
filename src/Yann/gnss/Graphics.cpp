#include "Graphics.h"


Graphics::Graphics(){}


void Graphics::addData(std::vector<double> X, std::vector<double> Y){
   addData(X, Y, Color());
}

void Graphics::addData(std::vector<double> X, std::vector<double> Y, Color color){
    addData(X, Y, color, 2);
}

void Graphics::addData(std::vector<double> X, std::vector<double> Y, Color color, int line_size){
    this->Xs.push_back(X);
    this->Ys.push_back(Y);
    this->S.push_back(line_size);
    this->C.push_back(color);
}

void Graphics::abline(double a, double b, double c){
    abline(a, b, c, Color());
}

void Graphics::abline(double a, double b, double c, Color color){
    abline(a, b, c, color, 2);
}

void Graphics::abline(double a, double b, double c, Color color, int line_size){
   std::vector<double> line;
   line.push_back(a); line.push_back(b); line.push_back(c);
   this->L.push_back(line);
   this->CL.push_back(color);
   this->SL.push_back(line_size);
}



// ---------------------------------------------------------------
// Print Plot data into SVG file
// ---------------------------------------------------------------
void Plot::writeToSVG(std::string path){

    std::ofstream output;
    output.open(path);

    output << "<!DOCTYPE html>\n";
    output << "<html>\n";
    output << "    <body>\n";

    for (unsigned i=0; i<this->G.size(); i++){
        output << this->G.at(i).printAsSVG();
    }

    // ------------------------------------------------
    // End of SVG file
    // ------------------------------------------------
    output << "    </body>\n";
    output << "</html>\n";

    output.close();

	std::cout << "Printing SVG file [" << path << "]... done                       " << std::endl;

}


// ---------------------------------------------------------------
// Print graphics data into SVG file
// ---------------------------------------------------------------
std::string Graphics::printAsSVG(){

    std::stringstream output;

    output << "        <svg height=\"" << (this->sy+100) << "\" width=\"" << (this->sx+100) << "\">\n";

    if (this->ticks){
        output << "            <style>\n";
        output << "                .small { font: italic " << this->tick_size << "px sans-serif; }\n";
        output << "                .heavy { font: bold " << this->label_size << "px sans-serif; }\n";
        output << "            </style>\n";
    }

    // ------------------------------------------------
    // Frame and ticks
    // ------------------------------------------------
    output << "            <polygon points=\"0,0 "<< this->sx << ",0 " << this->sx << "," << this->sy << " 0," << this->sy << " 0,0";
    output << "\"\n";
    output << "            style=\"fill:rgb(" << this->background.r << "," <<  this->background.g << "," <<  this->background.b << ");stroke:black;stroke-width:2\"/>\n";

    if (this->grid){
        for (double gx=this->xmin; gx<this->xmax; gx+=this->grid_rx){
            int x1 = this->x_px(gx);
            int y1 = this->sy;
            int y2 = 0;
            output << "            <line stroke-dasharray=\""<< this->grid_dash_type[0] << "," << this->grid_dash_type[1] << "\"";
            output << " x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x1 << "\" y2=\"" << y2 << "\"";
            output << "style=\"stroke:rgb(" << this->grid_color.r << "," <<  this->grid_color.g << "," <<  this->grid_color.b << ");";
            output << "stroke-width:" << this->grid_size << "\"/>\n";

        }
        for (double gy=this->ymin; gy<this->ymax; gy+=this->grid_ry){
            int y1 = this->y_px(gy);
            int x1 = this->sx;
            int x2 = 0;
            output << "            <line stroke-dasharray=\""<< this->grid_dash_type[0] << "," << this->grid_dash_type[1] << "\"";
            output << " x1=\"" << x1 << "\" y1=\"" << y1 << "\" x2=\"" << x2 << "\" y2=\"" << y1 << "\"";
            output << "style=\"stroke:rgb(" << this->grid_color.r << "," <<  this->grid_color.g << "," <<  this->grid_color.b << ");";
            output << "stroke-width:" << this->grid_size << "\"/>\n";
        }
    }

    // ------------------------------------------------
    // Data to plot
    // ------------------------------------------------

    // Curves
    for (unsigned i=0; i<this->Xs.size(); i++){

        std::vector<double> X = this->Xs.at(i);
        std::vector<double> Y = this->Ys.at(i);

        output << "            <polyline points=\"";

        for (unsigned j=0; j<X.size(); j++){
            if ((X.at(j) < this->xmin) || (X.at(j) > this->xmax)) continue;
            if ((Y.at(j) < this->ymin) || (Y.at(j) > this->ymax)) continue;
            output << this->x_px(X.at(j)) << "," << this->y_px(Y.at(j)) << " ";
        }

        output << "\"\n";
        output << "            style=\"fill:none;stroke:" << this->C.at(i).asString() << ";stroke-width:" << this->S.at(i) << "\"/>\n";

    }

    // Lines
    for (unsigned i=0; i<this->L.size(); i++){
        std::vector<double> line = this->L.at(i);
        double a = line.at(0);
        double b = line.at(1);
        double c = line.at(2);
        double x1 = 0;
        double x2 = 0;
        double y1 = 0;
        double y2 = 0;
        int xpx1, xpx2, ypx1, ypx2;

        if (b != 0){
            x1 = this->xmin; y1 = (-a*x1-c)/b;
            x2 = this->xmax; y2 = (-a*x2-c)/b;
        }else{
            y1 = this->ymin; x1 = -c/a;
            y2 = this->ymax; x2 = -c/a;
        }
        xpx1 = this->x_px(x1);
        xpx2 = this->x_px(x2);
        ypx1 = this->y_px(y1);
        ypx2 = this->y_px(y2);

        output << "            <line \"";
        output << " x1=\"" << xpx1 << "\" y1=\"" << ypx1 << "\" x2=\"" << xpx2 << "\" y2=\"" << ypx2 << "\"";
        output << "style=\"stroke:rgb(" << this->CL.at(i).r << "," <<  this->CL.at(i).g << "," <<  this->CL.at(i).b << ");";
        output << "stroke-width:" << this->SL.at(i) << "\"/>\n";
    }

    // ------------------------------------------------
    // Ticks and labels
    // ------------------------------------------------
    if (this->ticks){
        for (double gx=this->xmin; gx<this->xmax; gx+=this->grid_rx){
            int x1 = this->x_px(gx);
            int y1 = this->sy+20;
            output << "            <text x=\""<< x1 <<"\" y=\""<< y1 <<"\" class=\"small\">" << gx << "</text>\n";
        }
        for (double gy=this->ymin; gy<this->ymax; gy+=this->grid_ry){
            int x1 = this->sx+20;
            int y1 = this->y_px(gy);
            output << "            <text x=\""<< x1 <<"\" y=\""<< y1 <<"\" class=\"small\">" << gy << "</text>\n";
        }
    }
    if (this->labels){
        int x1 = this->sx/2.0 - 0.5*this->xlabel.size()*this->label_size/2;   int y1 = this->sy + 75;
        int y2 = this->sy/2.0 - 0.5*this->ylabel.size()*this->label_size/2;   int x2 = this->sx + 75;
        output << "            <text x=\""<< x1 <<"\" y=\""<< y1 <<"\" class=\"heavy\">" << this->xlabel << "</text>\n";
        output << "            <text x=\"0\" y=\"0\" class=\"heavy\" transform=\"translate(" << x2 << "," << y2 << ") rotate(90)\">" << this->ylabel << "</text>\n";
    }

    output << "        </svg>\n";

    return output.str();

}
