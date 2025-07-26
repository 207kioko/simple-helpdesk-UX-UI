from pysnmp.hlapi import SnmpEngine, CommunityData, UdpTransportTarget, ContextData, ObjectType, ObjectIdentity, getCmd



# 🖨️ Printer details
printer_ip = '192.168.2.45'
community = 'public'

# 📊 SNMP OIDs
TONER_OID = '1.3.6.1.2.1.43.11.1.1.9.1.1'  # Toner level
PAPER_OID = '1.3.6.1.2.1.43.8.2.1.10.1.1'  # Paper status

def get_snmp_value(ip, community, oid):
    iterator = getCmd(
        SnmpEngine(),
        CommunityData(community, mpModel=0),  # mpModel=0 = SNMPv1
        UdpTransportTarget((ip, 161), timeout=3, retries=2),
        ContextData(),
        ObjectType(ObjectIdentity(oid))
    )

    errorIndication, errorStatus, errorIndex, varBinds = next(iterator)

    if errorIndication:
        print(f"❌ SNMP Error: {errorIndication}")
        return None
    elif errorStatus:
        print(f"⚠️ SNMP Status Error: {errorStatus.prettyPrint()} at {errorIndex}")
        return None
    else:
        for varBind in varBinds:
            try:
                return int(varBind[1])
            except ValueError:
                print(f"Unexpected value type: {varBind[1]}")
                return None

def main():
    print(f"Checking printer at {printer_ip}...")

    toner_level = get_snmp_value(printer_ip, community, TONER_OID)
    paper_status = get_snmp_value(printer_ip, community, PAPER_OID)

    if toner_level is not None:
        print(f"✅ Toner Level: {toner_level}%")
    else:
        print("⚠️ Unable to fetch toner level.")

    if paper_status is not None:
        status_desc = {
            1: "Ready",
            2: "Low",
            3: "Empty"
        }.get(paper_status, "Unknown")
        print(f"✅ Paper Status: {status_desc} ({paper_status})")
    else:
        print("⚠️ Unable to fetch paper status.")

if __name__ == "__main__":
    main()
