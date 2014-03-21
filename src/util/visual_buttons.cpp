#if(ELISE_QT5)

#include "general/visual_buttons.h"

#include <iostream>


int imgListButton::buttonNumber=0;

imgListButton::imgListButton(const QString &text, QWidget *parent)
    : QPushButton(parent),
      unique_id(buttonNumber++)
{
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
    setText(text);
}

imgListButton::~imgListButton()
{

}

void imgListButton::onClick()
{
    //std::cout<<"bouton unique_id: "<<unique_id<<std::endl;
    emit my_click(unique_id);
}

 #endif // ELISE_QT5
