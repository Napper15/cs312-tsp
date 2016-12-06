using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NetworkRouting
{
    interface PriorityQueue
    {

        void makeQueue(int numberOfNodes);

        void insert(int index, double value);

        //Node deleteMin();

        int arrayDeleteMin();

        void decreaseKey(int index, double newKey);

        bool isEmpty();
    }
}
