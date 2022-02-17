
#include "MainWindow.h"
#include "NewProject.h"
#include "Tapioca.h"


newProject::newProject() : QWidget()
{
    setFixedSize(200, 100);



        m_lcd = new QLCDNumber(this);

        m_lcd->setSegmentStyle(QLCDNumber::Flat);

        m_lcd->move(50, 20);



        m_slider = new QSlider(Qt::Horizontal, this);

        m_slider->setGeometry(10, 60, 150, 20);




}








