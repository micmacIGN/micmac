#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>


class Color{

    public:

		Color(int r=0, int g=0, int b=0){
			this->r = rand() % 255;
			this->g = rand() % 255;
			this->b = rand() % 255;
		}

		Color(std::string name){
            if (name == "RED"){
                this->r = 255;
                this->g = 0;
                this->b = 0;
            }
            if (name == "GREEN"){
                this->r = 0;
                this->g = 170;
                this->b = 0;
            }
            if (name == "BLUE"){
                this->r = 0;
                this->g = 0;
                this->b = 255;
            }
            if (name == "WHITE"){
                this->r = 255;
                this->g = 255;
                this->b = 255;
            }
            if (name == "BLACK"){
                this->r = 0;
                this->g = 0;
                this->b = 0;
            }
		}

		int r;
		int g;
		int b;

		std::string asString(){return "rgb("+std::to_string(this->r)+","+std::to_string(this->g)+","+std::to_string(this->b)+")";}


};


class Graphics{

    public:

        Graphics();

        void addData(std::vector<double>, std::vector<double>);
        void addData(std::vector<double>, std::vector<double>, Color);
        void addData(std::vector<double>, std::vector<double>, Color, int);

        void abline(double, double, double);
        void abline(double, double, double, Color);
        void abline(double, double, double, Color, int);

        std::string printAsSVG();

        void setWidth(int sx){this->sx = sx;}
        void setHeight(int sy){this->sy = sy;}
        void setGridSize(int s){this->grid_size = s;}
        void setXmin(double xmin){this->xmin = xmin;}
        void setXmax(double xmax){this->xmax = xmax;}
        void setYmin(double ymin){this->ymin = ymin;}
        void setYmax(double ymax){this->ymax = ymax;}
        void setGridRx(double grid_rx){this->grid_rx = grid_rx;}
        void setGridRy(double grid_ry){this->grid_ry = grid_ry;}

        void setGridVisible(bool visible){this->grid = visible;}
        void setTicksVisible(bool visible){this->ticks = visible;}
        void setLabelsVisible(bool visible){this->labels = visible;}

        void setGridColor(Color color){this->grid_color = color;}
        void setBackgroundColor(Color color){this->background = color;}

        void setLabelSize(int label_size){this->label_size = label_size;}
        void setTickSize(int tick_size){this->tick_size = tick_size;}

        void setXLabel(std::string xlabel){this->xlabel = xlabel;}
        void setYLabel(std::string ylabel){this->ylabel = ylabel;}

        void setGridDashType(int s1, int s2){
            this->grid_dash_type[0] = s1;
            this->grid_dash_type[1] = s2;
        }

    private:

        int sx = 1000;
        int sy = 500;

        double xmin = 0;
        double xmax = 1;
        double ymin = 0;
        double ymax = 1;
        double grid_rx = 0.1;
        double grid_ry = 0.1;

        int label_size = 20;
        int tick_size = 15;

        bool grid = true;
        bool ticks = true;
        bool labels = true;

        Color grid_color = Color("BLACK");
        Color background = Color("WHITE");

        int grid_size = 2;
        int grid_dash_type[2] = {5,5};

        std::vector<std::vector<double>> Xs;
        std::vector<std::vector<double>> Ys;
        std::vector<Color> C;
        std::vector<int> S;

        std::vector<std::vector<double>> L;
        std::vector<Color> CL;
        std::vector<int> SL;

        std::string xlabel = "";
        std::string ylabel = "";

        int x_px(double x){return (x-this->xmin)/(this->xmax-this->xmin)*this->sx;}
        int y_px(double y){return (1-(y-this->ymin)/(this->ymax-this->ymin))*this->sy;}



};

class Plot{

    public:

        void addGraphics(Graphics graphics){this->G.push_back(graphics);}
        void writeToSVG(std::string path);

    private:

        std::vector<Graphics> G;


};

#endif // GRAPHICS_H
