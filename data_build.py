"""
Perspective of the deficit agent:

General data (sure):
    season (winter, summer, mid)
    day of the week (workweek, saturday, sunday/holiday)
    day time (1-96, every 15 minutes)
    ?? "topology" of negotiation

"My grid" data (sure):
    My wind turbines current production (%)
    My PVs -    -   -   -   -   -   -   (%)
    My loads

"Other" grid data (estimations):
    forecasts for generators
    price for fuelled generators
    SoC's
    loads estimations

"Other" grid data (sure):
    topology of the other grid
    Pmax of generators, of BSS

Negotiation history and results (sure):
    sent requests
    price-curves received
    negotiation iterations results

"""