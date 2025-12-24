### Component
Eveything is component. E.g. button, layout, group of buttons, full-page itself also if a component etc. all shoould be called as component. 

One component can be made up of one or more sub components.

### Components State
As there is hyricial dependency of the state of one componetent to another compontent.

There are two type of state of a compotent:
- rendering
- rendered
- error while fetching 

Once the compontent is rendered, then that compotenet have two new states of interaction:
- enabled
- disabled

All the states changes should be communciated with vizual clues and clear to the users and must be handled. 

### Principals
If we rendered the compotents, then we must make sure that the compotents layout does not keep changing ever now or then. We should and must follow the layout and once that layout is places it should keep remaing still.

There are layout rules:
- Layout per page.
- Layout of sub compotent in that page. 
-- Layout of the sub compotent (sub-compotent).

I don't think hidding the button once render it good approach instead it could be disabled is better approach. 

So, having the matrix of components and their states is good idea, for each page this way we can clearly define the states of the components and their interactions across various states.