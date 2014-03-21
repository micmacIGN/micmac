#ifndef VISUAL_BUTTONS_H
#define VISUAL_BUTTONS_H

#if(ELISE_QT5)

#include <QPushButton>

class imgListButton: public QPushButton
{
    Q_OBJECT

public:
        imgListButton(const QString &text, QWidget *parent=0);
        virtual ~imgListButton();

        QAction *Button;

protected:
        int unique_id;
        static int buttonNumber;

public slots:
        void onClick();

signals:
        void my_click(int);
};

#endif //ELISE_QT5

#endif /* VISUAL_BUTTONS_H */
