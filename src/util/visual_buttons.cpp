#if(ELISE_QT5)

#include "general/visual_buttons.h"

int selectionButton::buttonNumber=0;

selectionButton::selectionButton(QString text, QWidget *parent)
    : QPushButton(parent),
      unique_id(buttonNumber++)
{
    setText(text);
    connect(this,SIGNAL(clicked()),this,SLOT(onClick()));
}

void selectionButton::onClick()
{
    emit my_click(unique_id);
}

 #endif // ELISE_QT5
