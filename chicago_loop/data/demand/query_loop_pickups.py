import urllib.parse, urllib.request, json, sys
ds, tag = sys.argv[1], sys.argv[2]   # dataset id, label
where=("trip_start_timestamp >= '2023-04-05T00:00:00' AND trip_start_timestamp < '2023-04-06T00:00:00' "
       "AND pickup_community_area = 32")
q=urllib.parse.urlencode({"$select":"date_extract_hh(trip_start_timestamp) as h, count(*) as n",
  "$where":where, "$group":"h", "$order":"h"})
r=json.load(urllib.request.urlopen(f"https://data.cityofchicago.org/resource/{ds}.json?"+q,timeout=900))
json.dump(r,open(f'pickups_{tag}.json','w'))
print(tag, {int(x['h']):int(x['n']) for x in r})
