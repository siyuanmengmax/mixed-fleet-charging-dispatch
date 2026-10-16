import urllib.parse, urllib.request, json, sys
m=sys.argv[1]
where=(f"trip_start_timestamp >= '2023-{m}-01T00:00:00' AND trip_start_timestamp < '2023-{m}-15T00:00:00' "
       "AND pickup_community_area = 32 AND dropoff_community_area = 32 "
       "AND trip_seconds > 60 AND trip_seconds < 3600 AND trip_miles > 0.2 AND trip_miles < 5")
q=urllib.parse.urlencode({"$select":"date_extract_hh(trip_start_timestamp) as h, sum(trip_miles) as mi, sum(trip_seconds) as s, count(*) as n",
  "$where":where, "$group":"h", "$order":"h"})
r=json.load(urllib.request.urlopen("https://data.cityofchicago.org/resource/wrvz-psew.json?"+q,timeout=600))
json.dump(r,open(f'loop_speed_2023_{m}.json','w'))
for x in r:
    mi,s,n=float(x['mi']),float(x['s']),int(x['n'])
    print(x['h'], n, round(mi/n,2),'mi', round(mi*1.609/(s/3600),1),'km/h')
